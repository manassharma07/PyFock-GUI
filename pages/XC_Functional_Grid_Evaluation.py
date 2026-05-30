import os
import tempfile
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from ase.io import read as ase_read
from opt_einsum import contract
from gui_common import render_sidebar

try:
    import numba
except Exception:
    numba = None

try:
    import py3Dmol
except Exception:
    py3Dmol = None

try:
    import pylibxc
except Exception:
    pylibxc = None

from pyfock import Basis, DFT, Grids, Integrals, Mol, XC


BOHR_TO_ANGSTROM = 0.529177210903
DENSITY_PRUNE_THRESHOLD = 1.0e-8
DEFAULT_NCORES = 2

EXAMPLE_MOLECULES = {
    "Hydrogen": """2
Hydrogen
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.741400""",
    "Water": """3
Water
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161""",
    "Methane": """5
Methane
C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H -0.629118  0.629118 -0.629118
H  0.629118 -0.629118 -0.629118""",
    "Ammonia": """4
Ammonia
N  0.000000  0.000000  0.100000
H  0.945000  0.000000 -0.266000
H -0.472500  0.818000 -0.266000
H -0.472500 -0.818000 -0.266000""",
    "Carbon dioxide": """3
Carbon dioxide
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.160000
O  0.000000  0.000000 -1.160000""",
    "Formaldehyde": """4
Formaldehyde
C  0.000000  0.000000  0.000000
O  1.200000  0.000000  0.000000
H -0.550000  0.940000  0.000000
H -0.550000 -0.940000  0.000000""",
}

BASIS_SETS = ["sto-3g", "sto-6g", "3-21G", "6-31G", "def2-SVP"]

FUNCTIONALS = {
    "LDA exchange (LDA_X)": [1],
    "LDA exchange + PW correlation": [1, 12],
    "LDA exchange + VWN correlation": [1, 7],
    "PBE exchange (GGA_X_PBE)": [101],
    "PBE exchange + correlation": [101, 130],
    "PBESOL exchange + correlation": [116, 133],
    "RPBE exchange": [117],
    "BLYP exchange + correlation": [106, 131],
    "BP86 exchange + correlation": [106, 132],
    "TPSS exchange + correlation": [202, 231],
    "M06-L exchange + correlation": [203, 233],
    "r2SCAN exchange + correlation": [497, 498],
}

FAMILY_LABELS = {1: "LDA", 2: "GGA", 3: "Hybrid", 4: "mGGA"}


st.set_page_config(
    page_title="PyFock GUI - XC Grid Evaluation",
    layout="wide",
    page_icon="⚛️",
    menu_items={
        "About": "Evaluate PyFock XC energy densities and potentials on molecular grids."
    },
)

render_sidebar()


def _parse_xyz_to_atoms(xyz_text):
    return ase_read(StringIO(xyz_text), format="xyz")


def _structure_html(xyz_text, width=430, height=360):
    if py3Dmol is None:
        return None
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_text, "xyz")
    view.setStyle({"stick": {"radius": 0.18}, "sphere": {"scale": 0.30}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.show()
    view.render()
    js = view.js()
    return f"{js.startjs}{js.endjs}"


def _write_xyz_tempfile(xyz_text):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False)
    try:
        tmp.write(xyz_text)
        return tmp.name
    finally:
        tmp.close()


def _families_for_ids(func_ids):
    families = []
    for func_id in func_ids:
        families.append(XC.get_family(func_id))
    return families


def _as_1d(value):
    array = np.asarray(value, dtype=float)
    return np.ravel(array)


def _zeros_like(rho):
    return np.zeros_like(np.asarray(rho, dtype=float))


def _accumulate_result(total, result):
    names = ["zk", "vrho", "vsigma", "vtau"]
    for index, name in enumerate(names):
        if len(result) <= index:
            continue
        value = _as_1d(result[index])
        if name not in total:
            total[name] = value.copy()
        else:
            total[name] = total[name] + value
    return total


def _compute_pyfock_xc(func_ids, rho, sigma=None, tau=None):
    total = {}
    for func_id in func_ids:
        family = XC.get_family(func_id)
        kwargs = {"use_gpu": False}
        if family in (2, 4):
            kwargs["sigma"] = sigma
        if family == 4:
            kwargs["tau"] = tau
        result = XC.func_compute(func_id, rho, **kwargs)
        total = _accumulate_result(total, result)
    return total


def _compute_libxc_xc(func_ids, rho, sigma=None, tau=None):
    if pylibxc is None:
        return None

    total = {}
    for func_id in func_ids:
        family = XC.get_family(func_id)
        inp = {"rho": rho}
        if family in (2, 4):
            inp["sigma"] = sigma
        if family == 4:
            inp["tau"] = tau

        functional = pylibxc.LibXCFunctional(func_id, "unpolarized")
        result = functional.compute(inp)
        if "zk" in result:
            total["zk"] = total.get("zk", _zeros_like(rho)) + _as_1d(result["zk"])
        if "vrho" in result:
            total["vrho"] = total.get("vrho", _zeros_like(rho)) + _as_1d(result["vrho"])
        if "vsigma" in result:
            total["vsigma"] = total.get("vsigma", _zeros_like(rho)) + _as_1d(result["vsigma"])
        if "vtau" in result:
            total["vtau"] = total.get("vtau", _zeros_like(rho)) + _as_1d(result["vtau"])
    return total


def _comparison_stats(pyfock_result, libxc_result, mask=None, keys=None):
    if libxc_result is None:
        return pd.DataFrame()
    rows = []
    if mask is None:
        mask = slice(None)
    labels = [
        ("zk", "Energy density"),
        ("vrho", "Potential dE/drho"),
        ("vsigma", "Potential dE/dsigma"),
        ("vtau", "Potential dE/dtau"),
    ]
    if keys is not None:
        labels = [(key, label) for key, label in labels if key in keys]
    for key, label in labels:
        if key not in pyfock_result or key not in libxc_result:
            continue
        delta = np.asarray(pyfock_result[key])[mask] - np.asarray(libxc_result[key])[mask]
        if delta.size == 0:
            continue
        rows.append(
            {
                "Quantity": label,
                "Max abs error": float(np.max(np.abs(delta))),
                "Mean abs error": float(np.mean(np.abs(delta))),
                "RMS error": float(np.sqrt(np.mean(delta * delta))),
            }
        )
    return pd.DataFrame(rows)


def _example_script_text(functional_label, basis_name, grid_level, ncores, max_points):
    func_ids = FUNCTIONALS[functional_label]
    func_ids_text = ", ".join(str(func_id) for func_id in func_ids)
    return f'''import os
import tempfile

import numpy as np
from opt_einsum import contract
from pyfock import Basis, DFT, Grids, Integrals, Mol, XC

try:
    import pylibxc
except Exception:
    pylibxc = None

xyz = """{EXAMPLE_MOLECULES["Water"]}"""

with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as handle:
    handle.write(xyz)
    coordfile = handle.name

try:
    mol = Mol(coordfile=coordfile)
    basis = Basis(mol, {{"all": Basis.load(mol=mol, basis_name="{basis_name}")}})
    grids = Grids(mol, basis=basis, level={int(grid_level)}, ncores={int(ncores)})

    # Use a sample for quick demos. Remove this line to evaluate every grid point.
    idx = np.linspace(0, grids.coords.shape[0] - 1, {int(max_points)}, dtype=int)
    coords = grids.coords[idx]

    dft = DFT(mol, basis)
    dft.use_gpu = False
    dmat = dft.guessCoreH()

    func_ids = [{func_ids_text}]
    families = [XC.get_family(func_id) for func_id in func_ids]
    needs_sigma = any(family in (2, 4) for family in families)
    needs_tau = any(family == 4 for family in families)

    if needs_sigma:
        ao_values, ao_grad_values = Integrals.bf_val_helpers.eval_bfs_and_grad(
            basis, coords, parallel=True
        )
    else:
        ao_values = Integrals.bf_val_helpers.eval_bfs(basis, coords, parallel=True)
        ao_grad_values = None

    rho = Integrals.bf_val_helpers.eval_rho(ao_values, dmat)
    rho = np.maximum(rho, 1.0e-14)

    density_mask = rho > {DENSITY_PRUNE_THRESHOLD:.1e}
    coords = coords[density_mask]
    rho = rho[density_mask]
    if needs_sigma:
        ao_values = ao_values[density_mask]
        ao_grad_values = ao_grad_values[:, density_mask, :]

    sigma = None
    tau = None
    if needs_sigma:
        grad_x = contract("ij,mi,mj->m", dmat, ao_grad_values[0], ao_values)
        grad_x += contract("ij,mi,mj->m", dmat, ao_values, ao_grad_values[0])
        grad_y = contract("ij,mi,mj->m", dmat, ao_grad_values[1], ao_values)
        grad_y += contract("ij,mi,mj->m", dmat, ao_values, ao_grad_values[1])
        grad_z = contract("ij,mi,mj->m", dmat, ao_grad_values[2], ao_values)
        grad_z += contract("ij,mi,mj->m", dmat, ao_values, ao_grad_values[2])
        sigma = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z

    if needs_tau:
        tau = contract("ij,mi,mj->m", dmat, ao_grad_values[0], ao_grad_values[0])
        tau += contract("ij,mi,mj->m", dmat, ao_grad_values[1], ao_grad_values[1])
        tau += contract("ij,mi,mj->m", dmat, ao_grad_values[2], ao_grad_values[2])
        tau = np.maximum(0.5 * tau, 1.0e-14)

    pyfock = {{}}
    libxc = {{}} if pylibxc is not None else None
    for func_id, family in zip(func_ids, families):
        kwargs = {{"use_gpu": False}}
        libxc_input = {{"rho": rho}}
        if family in (2, 4):
            kwargs["sigma"] = sigma
            libxc_input["sigma"] = sigma
        if family == 4:
            kwargs["tau"] = tau
            libxc_input["tau"] = tau

        py_out = XC.func_compute(func_id, rho, **kwargs)
        for key, value in zip(["zk", "vrho", "vsigma", "vtau"], py_out):
            pyfock[key] = pyfock.get(key, 0.0) + np.ravel(value)

        if pylibxc is not None:
            lx_out = pylibxc.LibXCFunctional(func_id, "unpolarized").compute(libxc_input)
            for key in ["zk", "vrho", "vsigma", "vtau"]:
                if key in lx_out:
                    libxc[key] = libxc.get(key, 0.0) + np.ravel(lx_out[key])

    print("Grid points generated:", grids.coords.shape[0])
    print("Sampled grid points before density pruning:", idx.shape[0])
    print("Density pruning threshold:", {DENSITY_PRUNE_THRESHOLD:.1e})
    print("Grid points evaluated after density pruning:", coords.shape[0])
    print("PyFock energy density sample:", pyfock["zk"][:5])
    print("PyFock vrho sample:", pyfock["vrho"][:5])

    if libxc is not None:
        for key in sorted(set(pyfock) & set(libxc)):
            err = np.max(np.abs(pyfock[key] - libxc[key]))
            print(f"max |PyFock - LibXC| for {{key}}:", err)
    else:
        print("pylibxc is not installed; skipped LibXC comparison.")
finally:
    os.unlink(coordfile)
'''


def _sample_indices(n_points, max_points):
    if n_points <= max_points:
        return np.arange(n_points)
    return np.linspace(0, n_points - 1, max_points, dtype=int)


@st.cache_data(show_spinner=False)
def run_xc_grid_workflow(xyz_text, basis_name, grid_level, ncores, max_points, functional_label):
    if numba is not None:
        numba.set_num_threads(max(1, int(ncores)))

    coordfile = _write_xyz_tempfile(xyz_text)
    try:
        mol = Mol(coordfile=coordfile)
        basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
        grids = Grids(mol, basis=basis, level=int(grid_level), ncores=int(ncores))

        indices = _sample_indices(grids.coords.shape[0], int(max_points))
        coords = np.asarray(grids.coords[indices], dtype=float)
        weights = np.asarray(grids.weights[indices], dtype=float)

        func_ids = FUNCTIONALS[functional_label]
        families = _families_for_ids(func_ids)
        needs_sigma = any(family in (2, 4) for family in families)
        needs_tau = any(family == 4 for family in families)

        dft = DFT(mol, basis)
        dft.use_gpu = False
        dmat = dft.guessCoreH()

        if needs_sigma:
            ao_values, ao_grad_values = Integrals.bf_val_helpers.eval_bfs_and_grad(
                basis, coords, parallel=True
            )
        else:
            ao_values = Integrals.bf_val_helpers.eval_bfs(basis, coords, parallel=True)
            ao_grad_values = None

        rho = Integrals.bf_val_helpers.eval_rho(ao_values, dmat)
        rho = np.maximum(np.asarray(rho, dtype=float), 1.0e-14)
        n_grid_before_pruning = int(rho.shape[0])
        density_mask = rho > DENSITY_PRUNE_THRESHOLD
        n_grid_pruned = int(np.count_nonzero(density_mask))
        if n_grid_pruned == 0:
            raise ValueError(
                f"Density pruning removed every sampled grid point. "
                f"Increase the evaluation points or lower the threshold from {DENSITY_PRUNE_THRESHOLD:.1e}."
            )

        coords = coords[density_mask]
        weights = weights[density_mask]
        rho = rho[density_mask]
        ao_values = ao_values[density_mask]
        if ao_grad_values is not None:
            ao_grad_values = ao_grad_values[:, density_mask, :]

        sigma = None
        tau = None
        if needs_sigma:
            rho_grad_x = contract("ij,mi,mj->m", dmat, ao_grad_values[0], ao_values)
            rho_grad_x += contract("ij,mi,mj->m", dmat, ao_values, ao_grad_values[0])
            rho_grad_y = contract("ij,mi,mj->m", dmat, ao_grad_values[1], ao_values)
            rho_grad_y += contract("ij,mi,mj->m", dmat, ao_values, ao_grad_values[1])
            rho_grad_z = contract("ij,mi,mj->m", dmat, ao_grad_values[2], ao_values)
            rho_grad_z += contract("ij,mi,mj->m", dmat, ao_values, ao_grad_values[2])
            sigma = rho_grad_x * rho_grad_x + rho_grad_y * rho_grad_y + rho_grad_z * rho_grad_z
            sigma = np.maximum(np.asarray(sigma, dtype=float), 0.0)

        if needs_tau:
            tau = contract("ij,mi,mj->m", dmat, ao_grad_values[0], ao_grad_values[0])
            tau += contract("ij,mi,mj->m", dmat, ao_grad_values[1], ao_grad_values[1])
            tau += contract("ij,mi,mj->m", dmat, ao_grad_values[2], ao_grad_values[2])
            tau = np.maximum(0.5 * np.asarray(tau, dtype=float), 1.0e-14)

        pyfock_result = _compute_pyfock_xc(func_ids, rho, sigma=sigma, tau=tau)
        libxc_result = _compute_libxc_xc(func_ids, rho, sigma=sigma, tau=tau)
        assembled_result = None
        if libxc_result is not None:
            exc_pyfock, vxc_pyfock = Integrals.eval_xc_2(
                basis,
                dmat,
                weights,
                coords,
                func_ids,
                use_libxc=False,
                ncores=int(ncores),
                blocksize=5000,
                print_nelec=False,
            )
            exc_libxc, vxc_libxc = Integrals.eval_xc_2(
                basis,
                dmat,
                weights,
                coords,
                func_ids,
                use_libxc=True,
                ncores=int(ncores),
                blocksize=5000,
                print_nelec=False,
            )
            vxc_delta = np.asarray(vxc_pyfock) - np.asarray(vxc_libxc)
            assembled_result = {
                "PyFock Exc": float(exc_pyfock),
                "LibXC Exc": float(exc_libxc),
                "Abs Exc error": float(abs(exc_pyfock - exc_libxc)),
                "Max abs Vxc matrix error": float(np.max(np.abs(vxc_delta))),
                "RMS Vxc matrix error": float(np.sqrt(np.mean(vxc_delta * vxc_delta))),
            }

        return {
            "coords": coords,
            "weights": weights,
            "rho": rho,
            "sigma": sigma,
            "tau": tau,
            "pyfock": pyfock_result,
            "libxc": libxc_result,
            "assembled": assembled_result,
            "func_ids": func_ids,
            "families": [FAMILY_LABELS.get(family, str(family)) for family in families],
            "n_basis": int(basis.bfs_nao),
            "n_grid_total": int(grids.coords.shape[0]),
            "n_grid_before_pruning": n_grid_before_pruning,
            "n_grid_removed": int(n_grid_before_pruning - n_grid_pruned),
            "n_grid_eval": n_grid_pruned,
            "density_prune_threshold": DENSITY_PRUNE_THRESHOLD,
            "n_atoms": int(mol.natoms),
        }
    finally:
        try:
            os.unlink(coordfile)
        except OSError:
            pass


def _grid_figure(result, color_by, max_plot_points):
    coords_angstrom = result["coords"] * BOHR_TO_ANGSTROM
    n_points = coords_angstrom.shape[0]
    plot_idx = _sample_indices(n_points, max_plot_points)

    color_data = {
        "Density": result["rho"],
        "PyFock energy density": result["pyfock"].get("zk"),
        "PyFock potential dE/drho": result["pyfock"].get("vrho"),
        "Quadrature weight": result["weights"],
    }[color_by]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords_angstrom[plot_idx, 0],
                y=coords_angstrom[plot_idx, 1],
                z=coords_angstrom[plot_idx, 2],
                mode="markers",
                marker={
                    "size": 2.4,
                    "color": np.asarray(color_data)[plot_idx],
                    "colorscale": "Viridis",
                    "opacity": 0.68,
                    "colorbar": {"title": color_by},
                },
                text=[f"grid index {int(i)}" for i in plot_idx],
                hovertemplate=(
                    "x=%{x:.3f} A<br>y=%{y:.3f} A<br>z=%{z:.3f} A"
                    "<br>value=%{marker.color:.4e}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "b": 0, "t": 24},
        scene={
            "xaxis_title": "x (A)",
            "yaxis_title": "y (A)",
            "zaxis_title": "z (A)",
            "aspectmode": "data",
        },
    )
    return fig


def _comparison_figure(result, quantity):
    key_by_quantity = {
        "Energy density": "zk",
        "Potential dE/drho": "vrho",
        "Potential dE/dsigma": "vsigma",
        "Potential dE/dtau": "vtau",
    }
    key = key_by_quantity[quantity]
    py_vals = result["pyfock"][key]
    libxc_vals = result["libxc"][key]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=libxc_vals,
            y=py_vals,
            mode="markers",
            marker={"size": 5, "opacity": 0.55},
            name="Grid points",
        )
    )
    lo = float(min(np.min(libxc_vals), np.min(py_vals)))
    hi = float(max(np.max(libxc_vals), np.max(py_vals)))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "black"},
            name="y = x",
        )
    )
    fig.update_layout(
        height=430,
        xaxis_title=f"LibXC {quantity}",
        yaxis_title=f"PyFock {quantity}",
        margin={"l": 10, "r": 10, "b": 10, "t": 24},
    )
    return fig


def _parse_float_list(text, allow_zero=False):
    values = []
    for token in text.replace(",", " ").split():
        values.append(float(token))
    if not values:
        raise ValueError("Enter at least one density value.")
    array = np.asarray(values, dtype=float)
    if allow_zero:
        if np.any(array < 0):
            raise ValueError("Values must be non-negative.")
    elif np.any(array <= 0):
        raise ValueError("Values must be positive.")
    return array


def _tau_uniform_gas(rho):
    return 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)


def _direct_density_result(func_ids, rho, sigma=None, tau=None):
    pyfock_result = _compute_pyfock_xc(func_ids, rho, sigma=sigma, tau=tau)
    libxc_result = _compute_libxc_xc(func_ids, rho, sigma=sigma, tau=tau)
    table = pd.DataFrame({"rho": rho})
    if sigma is not None:
        table["sigma"] = sigma
    if tau is not None:
        table["tau"] = tau
    for key, label in [("zk", "zk"), ("vrho", "vrho"), ("vsigma", "vsigma"), ("vtau", "vtau")]:
        if key in pyfock_result:
            table[f"PyFock {label}"] = pyfock_result[key]
        if libxc_result is not None and key in libxc_result:
            table[f"LibXC {label}"] = libxc_result[key]
            table[f"Abs diff {label}"] = np.abs(pyfock_result[key] - libxc_result[key])
    stats = _comparison_stats(pyfock_result, libxc_result)
    return table, stats


def render_direct_density_samples():
    st.subheader("Direct Density Samples")
    st.write(
        "Evaluate XC values directly from density arrays. LDA needs only `rho`; "
        "GGA also needs `sigma`; meta-GGA also needs `tau`."
    )

    sample_functional = st.selectbox(
        "XC functional for density samples",
        list(FUNCTIONALS),
        index=1,
        key="density_sample_functional",
    )
    func_ids = FUNCTIONALS[sample_functional]
    families = _families_for_ids(func_ids)
    needs_sigma = any(family in (2, 4) for family in families)
    needs_tau = any(family == 4 for family in families)

    paste_tab, random_tab = st.tabs(["Paste densities", "Generate random densities"])
    with paste_tab:
        rho_text = st.text_area(
            "Density values",
            value="0.001 0.01 0.1 0.5 1.0 2.0",
            help="Use spaces, commas, or new lines.",
        )
        paste_rho = _parse_float_list(rho_text)

    with random_tab:
        cols = st.columns(4)
        n_samples = cols[0].number_input("Number", min_value=1, max_value=10000, value=64, step=1)
        rho_min_exp = cols[1].number_input("log10 min rho", value=-6.0, step=1.0)
        rho_max_exp = cols[2].number_input("log10 max rho", value=1.0, step=1.0)
        seed = cols[3].number_input("Seed", min_value=0, value=7, step=1)
        rng = np.random.default_rng(int(seed))
        random_rho = 10.0 ** rng.uniform(float(rho_min_exp), float(rho_max_exp), int(n_samples))

    source = st.radio(
        "Density source",
        ["Pasted values", "Random log-uniform"],
        horizontal=True,
        key="density_sample_source",
    )
    rho = paste_rho if source == "Pasted values" else random_rho

    sigma = None
    tau = None
    if needs_sigma:
        sigma_mode = st.selectbox(
            "Sigma input",
            ["zero gradient", "small random positive", "paste sigma values"],
            key="density_sample_sigma_mode",
        )
        if sigma_mode == "zero gradient":
            sigma = np.zeros_like(rho)
        elif sigma_mode == "small random positive":
            sigma = 1.0e-3 * rho ** (8.0 / 3.0)
        else:
            sigma_text = st.text_area(
                "Sigma values",
                value=" ".join(f"{x:.6e}" for x in np.zeros_like(rho)),
                key="density_sample_sigma_text",
            )
            sigma = _parse_float_list(sigma_text, allow_zero=True)
            if sigma.shape != rho.shape:
                st.error("Sigma must have the same number of values as rho.")
                return

    if needs_tau:
        tau_mode = st.selectbox(
            "Tau input",
            ["uniform electron gas estimate", "small random positive", "paste tau values"],
            key="density_sample_tau_mode",
        )
        if tau_mode == "uniform electron gas estimate":
            tau = _tau_uniform_gas(rho)
        elif tau_mode == "small random positive":
            tau = _tau_uniform_gas(rho) * 1.05
        else:
            tau_text = st.text_area(
                "Tau values",
                value=" ".join(f"{x:.6e}" for x in _tau_uniform_gas(rho)),
                key="density_sample_tau_text",
            )
            tau = _parse_float_list(tau_text, allow_zero=False)
            if tau.shape != rho.shape:
                st.error("Tau must have the same number of values as rho.")
                return

    if st.button("Evaluate density samples", type="primary", use_container_width=True):
        try:
            table, stats = _direct_density_result(func_ids, rho, sigma=sigma, tau=tau)
            st.dataframe(table, use_container_width=True)
            if pylibxc is not None:
                st.subheader("PyFock vs LibXC")
                st.dataframe(stats, use_container_width=True, hide_index=True)
            else:
                st.info("LibXC is not installed; showing PyFock values only.")
            st.download_button(
                "Download density sample values",
                table.to_csv(index=False).encode("utf-8"),
                file_name="pyfock_density_sample_xc_values.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.exception(exc)


st.title("XC Functional Evaluation on PyFock Grids")
st.markdown(
    """
Pick a small molecule, generate PyFock atom-centered grid points, build a core-Hamiltonian
density matrix, and evaluate XC energy density and potentials directly on those points.
"""
)

if pylibxc is None:
    st.info("LibXC Python bindings are not available, so the page will show PyFock-only results.")

left, right = st.columns([1.05, 1])

with left:
    st.subheader("Molecule")
    molecule_choice = st.selectbox("Small molecule", list(EXAMPLE_MOLECULES))
    xyz_content = st.text_area(
        "XYZ coordinates",
        value=EXAMPLE_MOLECULES[molecule_choice],
        height=190,
    )

with right:
    st.subheader("Preview")
    try:
        atoms = _parse_xyz_to_atoms(xyz_content)
        st.caption(f"{len(atoms)} atoms")
        html = _structure_html(xyz_content)
        if html:
            components.html(html, height=375)
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Element": atoms.get_chemical_symbols(),
                        "x": atoms.positions[:, 0],
                        "y": atoms.positions[:, 1],
                        "z": atoms.positions[:, 2],
                    }
                ),
                use_container_width=True,
            )
    except Exception as exc:
        st.warning(f"Could not parse XYZ coordinates: {exc}")

st.markdown("---")

settings_cols = st.columns([1, 1, 1])
with settings_cols[0]:
    basis_name = st.selectbox("Basis set", BASIS_SETS, index=0)
with settings_cols[1]:
    grid_level = st.slider("Grid level", 3, 6, 3)
with settings_cols[2]:
    max_points = st.select_slider(
        "Evaluation points",
        options=[250, 500, 1000, 2000, 4000],
        value=1000,
    )
functional_label = st.selectbox("XC functional", list(FUNCTIONALS), index=1)
ncores = DEFAULT_NCORES

run_clicked = st.button("Generate grid and evaluate XC", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Generating grids, building the core-H guess, and evaluating XC values..."):
        try:
            st.session_state.xc_grid_result = run_xc_grid_workflow(
                xyz_content,
                basis_name,
                grid_level,
                ncores,
                max_points,
                functional_label,
            )
            st.session_state.xc_grid_functional_label = functional_label
        except Exception as exc:
            st.session_state.xc_grid_result = None
            st.exception(exc)

result = st.session_state.get("xc_grid_result")

if result:
    st.markdown("---")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Atoms", result["n_atoms"])
    metric_cols[1].metric("Basis functions", result["n_basis"])
    metric_cols[2].metric("Generated grid points", f"{result['n_grid_total']:,}")
    metric_cols[3].metric("Sampled points", f"{result['n_grid_before_pruning']:,}")
    metric_cols[4].metric("Evaluated points", f"{result['n_grid_eval']:,}")
    metric_cols[5].metric("LibXC", "Available" if result["libxc"] is not None else "Not available")

    st.caption(
        "Functional IDs: "
        + ", ".join(str(fid) for fid in result["func_ids"])
        + " | Families: "
        + ", ".join(result["families"])
    )
    st.info(
        f"Density pruning is applied before XC evaluation: only grid points with "
        f"rho > {result['density_prune_threshold']:.1e} are evaluated. "
        f"Kept {result['n_grid_eval']:,} of {result['n_grid_before_pruning']:,} sampled points "
        f"and removed {result['n_grid_removed']:,} low-density tail points."
    )

    grid_col, table_col = st.columns([1.45, 1])
    with grid_col:
        color_by = st.selectbox(
            "Color grid by",
            ["Density", "PyFock energy density", "PyFock potential dE/drho", "Quadrature weight"],
        )
        max_plot_points = st.slider(
            "Plotted points",
            min_value=1,
            max_value=max(1, result["n_grid_eval"]),
            value=min(4000, max(1, result["n_grid_eval"])),
            step=1,
        )
        st.plotly_chart(_grid_figure(result, color_by, max_plot_points), use_container_width=True)

    with table_col:
        st.subheader("Grid Values")
        table = pd.DataFrame(
            {
                "x (Bohr)": result["coords"][:, 0],
                "y (Bohr)": result["coords"][:, 1],
                "z (Bohr)": result["coords"][:, 2],
                "weight": result["weights"],
                "rho": result["rho"],
                "PyFock zk": result["pyfock"].get("zk"),
                "PyFock vrho": result["pyfock"].get("vrho"),
            }
        )
        if result["sigma"] is not None:
            table["sigma"] = result["sigma"]
        if result["tau"] is not None:
            table["tau"] = result["tau"]
        st.dataframe(table.head(250), use_container_width=True, height=445)
        st.download_button(
            "Download evaluated grid values",
            table.to_csv(index=False).encode("utf-8"),
            file_name="pyfock_xc_grid_values.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if result["libxc"] is not None:
        st.markdown("---")
        st.subheader("PyFock vs LibXC")
        if result.get("assembled") is not None:
            st.caption("Production-style assembled XC energy and matrix check")
            st.dataframe(
                pd.DataFrame([result["assembled"]]),
                use_container_width=True,
                hide_index=True,
            )

        is_mgga = any(family == "mGGA" for family in result["families"])
        main_keys = ["zk", "vrho"] if is_mgga else None
        stats = _comparison_stats(result["pyfock"], result["libxc"], keys=main_keys)
        st.caption(
            f"Pointwise comparison on density-pruned grid points "
            f"(rho > {result['density_prune_threshold']:.1e})"
        )
        st.dataframe(stats, use_container_width=True, hide_index=True)
        if is_mgga:
            with st.expander("Advanced raw meta-GGA derivative diagnostics"):
                st.write(
                    "Raw pointwise `vsigma` and `vtau` are shown here only as numerical "
                    "diagnostics. Meta-GGA expressions contain branch points and clipping, "
                    "so finite-difference derivative channels can look much noisier than "
                    "the assembled XC energy and matrix used by DFT."
                )
                st.dataframe(
                    _comparison_stats(result["pyfock"], result["libxc"], keys=["vsigma", "vtau"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Why density pruning is used"):
            st.write(
                "For GGA and meta-GGA functionals, `sigma` is the density-gradient invariant "
                "`grad(rho) dot grad(rho)`. In the far tail of an atom-centered grid, both `rho` "
                "and `sigma` can be extremely close to zero. The derivative with respect to "
                "`sigma` is then poorly conditioned, so tiny implementation differences, cutoffs, "
                "or regularization choices between PyFock and LibXC can turn into very large "
                "absolute `vsigma` differences. This page therefore prunes low-density tail "
                "points before evaluating PyFock or LibXC. For meta-GGA functionals, `vsigma` "
                "and `vtau` are especially sensitive because they depend on both density-gradient "
                "and kinetic-energy-density inputs; treat those pointwise derivative comparisons "
                "as diagnostics rather than integrated energy validation. The assembled XC check "
                "above is closer to what the DFT calculation actually uses."
            )

        comparable = []
        plot_candidates = [
            ("Energy density", "zk"),
            ("Potential dE/drho", "vrho"),
            ("Potential dE/dsigma", "vsigma"),
            ("Potential dE/dtau", "vtau"),
        ]
        if is_mgga:
            plot_candidates = plot_candidates[:2]
        for label, key in plot_candidates:
            if key in result["pyfock"] and key in result["libxc"]:
                comparable.append(label)

        if comparable:
            quantity = st.selectbox("Comparison plot quantity", comparable)
            st.plotly_chart(_comparison_figure(result, quantity), use_container_width=True)
    else:
        st.markdown("---")
        st.subheader("PyFock Results")
        st.write(
            "LibXC is not installed in this environment. The generated grid, density, "
            "energy density, and potential values above are still available from PyFock."
        )
else:
    st.info("Choose a molecule and settings, then run the evaluation.")

# st.markdown("---")
# render_direct_density_samples()

st.markdown("---")
with st.expander("Minimal Python example script"):
    st.code(
        _example_script_text(functional_label, basis_name, grid_level, ncores, max_points),
        language="python",
    )
