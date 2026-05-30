from collections import Counter
from io import StringIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from gui_common import render_sidebar

try:
    import py3Dmol
except Exception:
    py3Dmol = None

try:
    from ase.io import read as ase_read
except Exception:
    ase_read = None

from pyfock import Basis, Mol


REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = Path(__file__).resolve().parents[1]
STRUCTURES_DIR = GUI_ROOT / "structures"

BASIS_SETS = [
    "sto-3g",
    "sto-6g",
    "3-21G",
    "6-31G",
    "6-31+G",
    "6-31++G",
    "6-311G",
    "cc-pVDZ",
    "cc-pVTZ",
    "aug-cc-pVDZ",
    "def2-SVP",
    "def2-SVPD",
    "def2-TZVP",
    "def2-TZVPD",
    "def2-QZVP",
    "def2-QZVPD",
]

SHELL_NAMES = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
    5: "h",
    6: "i",
}

MAX_TABLE_ROWS = 1000
LARGE_MOLECULE_WARNING_AT = 500


st.set_page_config(
    page_title="PyFock GUI - Basis Functions",
    layout="wide",
    page_icon=":atom_symbol:",
    menu_items={"About": "Explore PyFock basis set objects from molecules and XYZ input."},
)

render_sidebar()


def _parse_xyz_text(xyz_text):
    text = xyz_text.strip()
    if not text:
        raise ValueError("The XYZ input is empty.")

    if ase_read is not None:
        atoms = ase_read(StringIO(text), format="xyz")
        symbols = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        return [[symbols[i], float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])] for i in range(len(symbols))]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        natoms = int(lines[0])
        atom_lines = lines[2 : 2 + natoms]
    except Exception:
        atom_lines = lines

    atoms = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Could not parse XYZ line: {line}")
        atoms.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms


def _atoms_to_xyz_text(atoms, title="Custom molecule"):
    lines = [str(len(atoms)), title]
    for symbol, x, y, z in atoms:
        lines.append(f"{symbol:2s} {float(x): .10f} {float(y): .10f} {float(z): .10f}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def _load_benchmark_xyzs():
    molecules = []
    for path in sorted(STRUCTURES_DIR.glob("*.xyz")):
        try:
            text = path.read_text()
            first_line = text.splitlines()[0].strip()
            natoms = int(first_line)
        except Exception:
            continue
        molecules.append(
            {
                "name": path.stem.replace("_", " "),
                "filename": path.name,
                "path": str(path),
                "natoms": natoms,
                "text": text,
            }
        )
    return molecules


def _structure_html(xyz_text, width=460, height=360):
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


def _safe_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _fmt_tuple(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(str(x) for x in value) + ")"
    return str(value)


def _download_dataframe(df, filename, label):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _show_dataframe(df, filename, label, height=360):
    if len(df) > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS:,} of {len(df):,} rows. Download the CSV for the full table.")
        shown = df.head(MAX_TABLE_ROWS)
    else:
        shown = df
    st.dataframe(shown, use_container_width=True, height=height)
    _download_dataframe(df, filename, label)


def _basis_loadability_report(atoms, basis_name):
    missing = []
    for symbol in sorted({atom[0] for atom in atoms}):
        block = Basis.load(atom=symbol, basis_name=basis_name, quiet=True)
        if block.strip() == "":
            missing.append(symbol)
    return missing


@st.cache_data(show_spinner=False)
def _build_basis_payload(xyz_text, basis_name):
    atoms = _parse_xyz_text(xyz_text)
    mol = Mol(atoms=atoms)
    if not getattr(mol, "success", False):
        raise ValueError("PyFock could not initialize the molecule from this XYZ input.")

    missing = _basis_loadability_report(atoms, basis_name)
    if missing:
        raise ValueError(f"{basis_name} is not available for: {', '.join(missing)}")

    basis_string = Basis.load(mol=mol, basis_name=basis_name, quiet=True)
    basis = Basis(mol, {"all": basis_string})
    if int(getattr(basis, "bfs_nao", 0)) == 0:
        raise ValueError("PyFock could not build any basis functions for this molecule and basis set.")

    return _serialize_payload(mol, basis, atoms, basis_name)


def _serialize_payload(mol, basis, atoms, basis_name):
    symbols = list(mol.atomicSpecies)
    basis_species = list(getattr(mol, "basisSpecies", symbols))
    coords = np.array(mol.coords, dtype=float)
    atom_counts = Counter(symbols)
    shell_atom_indices = _shell_atom_indices(basis)

    shell_rows = []
    for shell_idx in range(int(basis.nshells)):
        atom_idx = int(shell_atom_indices[shell_idx]) if shell_idx < len(shell_atom_indices) else None
        l_value = int(basis.shells[shell_idx]) - 1
        coord = np.array(basis.shell_coords[shell_idx], dtype=float)
        prim_indices = _primitive_indices_for_shell(basis, shell_idx)
        shell_expnts = np.array([basis.prim_expnts[i] for i in prim_indices], dtype=float)
        alpha_min = np.min(shell_expnts)
        alpha_max = np.max(shell_expnts)
        shell_rows.append(
            {
                "shell_index": shell_idx,
                "atom_index": atom_idx,
                "atom": symbols[atom_idx] if atom_idx is not None and atom_idx < len(symbols) else "",
                "basis_species": basis_species[atom_idx] if atom_idx is not None and atom_idx < len(basis_species) else "",
                "shell_label": str(basis.shellsLabel[shell_idx]),
                "angular_l": l_value,
                "angular_name": SHELL_NAMES.get(l_value, f"l={l_value}"),
                "n_primitives": int(basis.nprims[shell_idx]),
                "n_basis_functions": int(basis.bfs_nbfshell[shell_idx]),
                "basis_function_offset": int(basis.shell_bfs_offset[shell_idx]),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
                "min_exponent": float(alpha_min),
                "max_exponent": float(alpha_max),
            }
        )

    bf_rows = []
    for bf_idx in range(int(basis.bfs_nao)):
        atom_idx = int(basis.bfs_atoms[bf_idx]) if len(basis.bfs_atoms) > bf_idx else None
        shell_idx = int(basis.bfs_shell_index[bf_idx])
        coord = np.array(basis.bfs_coords[bf_idx], dtype=float)
        lmn = _safe_list(basis.bfs_lmn[bf_idx])
        bf_rows.append(
            {
                "basis_function_index": bf_idx,
                "label": str(basis.bfs_label[bf_idx]),
                "atom_index": atom_idx,
                "atom": symbols[atom_idx] if atom_idx is not None and atom_idx < len(symbols) else "",
                "shell_index": shell_idx,
                "shell_label": str(basis.shellsLabel[shell_idx]),
                "angular_l": int(basis.bfs_lm[bf_idx]),
                "lmn": _fmt_tuple(lmn),
                "n_primitives": int(basis.bfs_nprim[bf_idx]),
                "contraction_norm": float(basis.bfs_contr_prim_norms[bf_idx]),
                "radius_cutoff": float(basis.bfs_radius_cutoff[bf_idx]),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
                "primitive_exponents": _fmt_tuple(basis.bfs_expnts[bf_idx]),
                "primitive_coefficients": _fmt_tuple(basis.bfs_coeffs[bf_idx]),
                "primitive_norms": _fmt_tuple(basis.bfs_prim_norms[bf_idx]),
            }
        )

    primitive_rows = []
    bfs_indices_by_shell = {}
    for bf_idx, shell_idx in enumerate(basis.bfs_shell_index):
        bfs_indices_by_shell.setdefault(int(shell_idx), []).append(bf_idx)

    for prim_idx in range(int(basis.totalnprims)):
        atom_idx = int(basis.prim_atoms[prim_idx])
        shell_idx = int(basis.prim_shells[prim_idx])
        coord = np.array(basis.prim_coords[prim_idx], dtype=float)
        shell_prim_indices = _primitive_indices_for_shell(basis, shell_idx)
        primitive_position_in_shell = shell_prim_indices.index(prim_idx)
        norms_by_basis_function = {}
        for bf_idx in bfs_indices_by_shell.get(shell_idx, []):
            if primitive_position_in_shell < len(basis.bfs_prim_norms[bf_idx]):
                norms_by_basis_function[str(basis.bfs_label[bf_idx])] = float(
                    basis.bfs_prim_norms[bf_idx][primitive_position_in_shell]
                )
        primitive_rows.append(
            {
                "primitive_index": prim_idx,
                "atom_index": atom_idx,
                "atom": symbols[atom_idx] if atom_idx < len(symbols) else "",
                "shell_index": shell_idx,
                "shell_label": str(basis.shellsLabel[shell_idx]),
                "angular_l": int(basis.shells[shell_idx]) - 1,
                "exponent_alpha": float(basis.prim_expnts[prim_idx]),
                "coefficient": float(basis.prim_coeffs[prim_idx]),
                "norms_by_basis_function": json.dumps(norms_by_basis_function),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
            }
        )

    atom_rows = []
    bf_count_by_atom = Counter(row["atom_index"] for row in bf_rows)
    shell_count_by_atom = Counter(row["atom_index"] for row in shell_rows)
    prim_count_by_atom = Counter(row["atom_index"] for row in primitive_rows)
    for i, symbol in enumerate(symbols):
        atom_rows.append(
            {
                "atom_index": i,
                "atom": symbol,
                "basis_species": basis_species[i],
                "Z": int(mol.Zcharges[i]),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "shells": int(shell_count_by_atom[i]),
                "basis_functions": int(bf_count_by_atom[i]),
                "primitives": int(prim_count_by_atom[i]),
                "max_exponent_for_atom": float(basis.alpha_max[i]) if i < len(basis.alpha_max) else np.nan,
                "min_exponents_by_l": json.dumps(_safe_dict(getattr(basis, "alpha_min", [{}])[i] if i < len(basis.alpha_min) else {})),
            }
        )

    ecp_rows = []
    for ecp in getattr(basis, "ecps", []):
        atom_idx = int(ecp.get("atom_index", -1))
        ecp_rows.append(
            {
                "atom_index": atom_idx,
                "atom": symbols[atom_idx] if 0 <= atom_idx < len(symbols) else "",
                "header": str(ecp.get("header", "")),
                "core_electrons": int(ecp.get("ncore", ecp.get("core_electrons", 0))),
                "lmax": int(ecp.get("lmax", 0)),
                "local_channel": str(ecp.get("local_label", "")),
                "local_terms": len(ecp.get("local_terms", [])),
                "projector_channels": len(ecp.get("projector_terms", {})),
                "projector_terms": sum(len(v) for v in ecp.get("projector_terms", {}).values()),
            }
        )

    raw_arrays = {
        "nprims": _safe_list(basis.nprims),
        "prim_atoms": _safe_list(basis.prim_atoms),
        "nprims_atoms": _safe_list(basis.nprims_atoms),
        "prim_shells": _safe_list(basis.prim_shells),
        "nprims_shells": _safe_list(basis.nprims_shells),
        "nprims_shell_l_list": _safe_list(basis.nprims_shell_l_list),
        "alpha_max": _safe_list(basis.alpha_max),
        "alpha_min": _safe_list(basis.alpha_min),
        "shellsLabel": _safe_list(basis.shellsLabel),
        "shells": _safe_list(basis.shells),
        "bfs_label": _safe_list(basis.bfs_label),
        "bfs_lmn": _safe_list(basis.bfs_lmn),
        "bfs_lm": _safe_list(basis.bfs_lm),
        "bfs_nbfshell": _safe_list(basis.bfs_nbfshell),
        "shell_bfs_offset": _safe_list(basis.shell_bfs_offset),
        "bfs_shell_index": _safe_list(basis.bfs_shell_index),
        "bfs_nprim": _safe_list(basis.bfs_nprim),
        "bfs_atoms": _safe_list(basis.bfs_atoms),
        "bfs_radius_cutoff": _safe_list(basis.bfs_radius_cutoff),
    }

    return {
        "basis_name": basis_name,
        "natoms": int(mol.natoms),
        "nelectrons": int(mol.nelectrons),
        "charge": int(mol.charge),
        "elements": dict(atom_counts),
        "bfs_nao": int(basis.bfs_nao),
        "nshells": int(basis.nshells),
        "totalnprims": int(basis.totalnprims),
        "has_ecp": bool(getattr(basis, "has_ecp", False)),
        "ecp_total_core_electrons": int(getattr(basis, "ecp_total_core_electrons", 0)),
        "basis_string": str(basis.basisSet),
        "atoms_df": pd.DataFrame(atom_rows),
        "shells_df": pd.DataFrame(shell_rows),
        "basis_functions_df": pd.DataFrame(bf_rows),
        "primitives_df": pd.DataFrame(primitive_rows),
        "ecp_df": pd.DataFrame(ecp_rows),
        "raw_arrays": raw_arrays,
    }


def _safe_dict(value):
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    return value


def _shell_atom_indices(basis):
    shell_atoms = []
    offset = 0
    for nprim in basis.nprims:
        if int(nprim) > 0 and offset < len(basis.prim_atoms):
            shell_atoms.append(int(basis.prim_atoms[offset]))
        else:
            shell_atoms.append(None)
        offset += int(nprim)
    return shell_atoms


def _primitive_indices_for_shell(basis, shell_idx):
    return [i for i, value in enumerate(basis.prim_shells) if int(value) == int(shell_idx)]


def _metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def _render_teaching_notes():
    st.markdown(
        """
        This page reads the same PyFock `Basis` object used by the integral and DFT code, but presents it in the order students usually learn it:
        atoms define centers, each center owns shells, each shell expands into Cartesian basis functions, and each basis function is a contracted sum of primitive Gaussians.

        A primitive Gaussian is controlled mostly by its exponent `alpha` and contraction coefficient. Large exponents are tight functions near the nucleus; small exponents are diffuse functions. The `lmn` triplet labels the Cartesian angular part, so `(1, 0, 0)` is `x`, `(0, 1, 0)` is `y`, and so on.
        """
    )


def _render_overview(payload):
    atoms_df = payload["atoms_df"]
    shells_df = payload["shells_df"]
    bf_df = payload["basis_functions_df"]

    metric_cols = st.columns(6)
    with metric_cols[0]:
        _metric("Atoms", f"{payload['natoms']:,}")
    with metric_cols[1]:
        _metric("Electrons", f"{payload['nelectrons']:,}")
    with metric_cols[2]:
        _metric("Shells", f"{payload['nshells']:,}")
    with metric_cols[3]:
        _metric("Basis functions", f"{payload['bfs_nao']:,}", "PyFock stores this as basis.bfs_nao.")
    with metric_cols[4]:
        _metric("Primitives", f"{payload['totalnprims']:,}")
    with metric_cols[5]:
        _metric("ECP core e-", f"{payload['ecp_total_core_electrons']:,}")

    st.markdown("#### Composition")
    c1, c2 = st.columns([1, 1])
    with c1:
        element_df = pd.DataFrame(
            [{"element": key, "count": value} for key, value in payload["elements"].items()]
        ).sort_values("element")
        st.dataframe(element_df, use_container_width=True, height=240)
    with c2:
        shell_counts = shells_df.groupby(["angular_name", "angular_l"], as_index=False).size()
        shell_counts = shell_counts.sort_values("angular_l")
        fig = px.bar(shell_counts, x="angular_name", y="size", labels={"size": "shell count", "angular_name": "shell"})
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Per-atom basis summary")
    _show_dataframe(atoms_df, "basis_atoms_summary.csv", "Download atom summary", height=360)

    st.markdown("#### Basis functions per atom")
    if len(bf_df) > 0:
        bf_by_atom = atoms_df[["atom_index", "atom", "basis_functions", "shells", "primitives"]]
        fig = px.bar(
            bf_by_atom,
            x="atom_index",
            y="basis_functions",
            color="atom",
            hover_data=["shells", "primitives"],
            labels={"atom_index": "atom index", "basis_functions": "basis functions"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def _render_shells(payload):
    st.markdown(
        "Each shell is one angular-momentum block on an atom. For example, a p shell has three Cartesian basis functions: px, py, and pz."
    )
    _show_dataframe(payload["shells_df"], "basis_shells.csv", "Download shell table", height=520)


def _render_basis_functions(payload):
    st.markdown(
        "A basis function is a contracted atomic orbital. PyFock stores its center, shell, angular label, primitive exponents, primitive coefficients, and normalization factors."
    )
    _show_dataframe(payload["basis_functions_df"], "basis_functions.csv", "Download basis function table", height=520)


def _render_primitives(payload):
    st.markdown(
        "Primitive Gaussians are the individual exponential functions. Contracted basis functions reuse the same primitive list with different angular labels inside a shell. The primitive normalization is therefore shown by basis-function label, because it depends on the Cartesian angular part."
    )
    primitives_df = payload["primitives_df"].drop(columns=["primitive_norm"], errors="ignore")
    _show_dataframe(primitives_df, "basis_primitives.csv", "Download primitive table", height=520)


def _render_ecp(payload):
    if not payload["has_ecp"]:
        st.info("No effective core potential data is stored for this basis object.")
        return
    st.markdown(
        "This basis object includes effective core potential data. PyFock removes the listed core electrons from the explicit electron count and stores local/projector ECP terms by atom."
    )
    _show_dataframe(payload["ecp_df"], "basis_ecp.csv", "Download ECP table", height=320)


def _render_raw(payload):
    st.markdown("These are the PyFock `Basis` arrays exposed almost directly. They are useful when connecting the teaching view back to code.")
    for key, value in payload["raw_arrays"].items():
        with st.expander(key):
            st.code(json.dumps(value, indent=2), language="json")
    with st.expander("Complete TURBOMOLE-format basis string"):
        st.code(payload["basis_string"], language="text")


st.title("Basis Function Explorer")
st.caption("Inspect PyFock basis objects as atoms, shells, contracted basis functions, primitives, and optional ECP data.")

benchmarks = _load_benchmark_xyzs()
if not benchmarks:
    st.error(f"No XYZ files were found in {STRUCTURES_DIR}.")
    st.stop()

source_col, basis_col, rows_col = st.columns([1.4, 1, 1])
with source_col:
    molecule_source = st.radio("Molecule input", ["Built-in XYZ", "Custom XYZ"], horizontal=True)
with basis_col:
    basis_name = st.selectbox("Basis set", BASIS_SETS, index=BASIS_SETS.index("def2-SVP"))
with rows_col:
    st.metric("Built-in structures", f"{len(benchmarks):,}")

if molecule_source == "Built-in XYZ":
    labels = [f"{item['name']} ({item['natoms']} atoms)" for item in benchmarks]
    default_index = next((i for i, item in enumerate(benchmarks) if item["filename"] == "H2O.xyz"), 0)
    selected_label = st.selectbox("Select molecule", labels, index=default_index)
    selected = benchmarks[labels.index(selected_label)]
    xyz_text = selected["text"]
    molecule_name = selected["name"]
else:
    default_xyz = """3
Water
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161"""
    xyz_text = st.text_area("Custom XYZ", default_xyz, height=220)
    molecule_name = "Custom molecule"

try:
    atoms_preview = _parse_xyz_text(xyz_text)
except Exception as exc:
    st.error(f"Could not parse the XYZ input: {exc}")
    st.stop()

natoms_preview = len(atoms_preview)
if natoms_preview > LARGE_MOLECULE_WARNING_AT:
    st.warning(
        f"This molecule has {natoms_preview:,} atoms. PyFock will still try to build the basis, but the full tables can be large."
    )

st.markdown("---")
preview_col, guide_col = st.columns([1, 1.3])
with preview_col:
    st.subheader(molecule_name)
    xyz_for_view = _atoms_to_xyz_text(atoms_preview, molecule_name)
    html = _structure_html(xyz_for_view)
    if html is not None:
        components.html(html, height=380)
    else:
        st.dataframe(
            pd.DataFrame(atoms_preview, columns=["atom", "x", "y", "z"]),
            use_container_width=True,
            height=320,
        )
with guide_col:
    st.subheader("What PyFock Stores")
    _render_teaching_notes()

run = st.button("Build basis object", type="primary", use_container_width=True)
if not run and "basis_payload" not in st.session_state:
    st.info("Choose a molecule and basis set, then build the basis object.")
    st.stop()

if run:
    with st.spinner("Building PyFock Basis object..."):
        try:
            st.session_state["basis_payload"] = _build_basis_payload(xyz_text, basis_name)
            st.session_state["basis_payload_key"] = (molecule_name, basis_name)
        except Exception as exc:
            st.error(f"Basis construction failed: {exc}")
            st.stop()

payload = st.session_state["basis_payload"]
payload_key = st.session_state.get("basis_payload_key", (molecule_name, basis_name))

st.success(
    f"Built {payload_key[1]} for {payload_key[0]}: {payload['bfs_nao']:,} basis functions, "
    f"{payload['nshells']:,} shells, {payload['totalnprims']:,} primitives."
)

tabs = st.tabs(["Overview", "Shells", "Basis functions", "Primitives", "ECP", "Raw arrays"])
with tabs[0]:
    _render_overview(payload)
with tabs[1]:
    _render_shells(payload)
with tabs[2]:
    _render_basis_functions(payload)
with tabs[3]:
    _render_primitives(payload)
with tabs[4]:
    _render_ecp(payload)
with tabs[5]:
    _render_raw(payload)
