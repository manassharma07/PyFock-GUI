"""Geometry optimization with PyFock forces and ASE optimizers."""

import io
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from ase.io import read as ase_read
from plotly.subplots import make_subplots

from geometry_optimization import (
    MAX_OPTIMIZATION_CYCLES,
    OPTIMIZER_HELP,
    OPTIMIZERS,
    atoms_to_xyz,
    build_reproduction_script,
    frames_to_extxyz,
    maximum_force,
    snapshot_with_results,
)
from gui_common import render_sidebar


st.set_page_config(
    page_title="Geometry Optimization - PyFock GUI",
    page_icon="📐",
    layout="wide",
    menu_items={
        "About": "Optimize molecular geometries with PyFock DFT forces and ASE."
    },
)
render_sidebar()


APP_ROOT = Path(__file__).resolve().parents[1]
GUI_GITHUB_URL = "https://github.com/manassharma07/PyFock-GUI"

XC_FUNCTIONALS = [
    "LDA",
    "HF",
    "PBE",
    "PBESOL",
    "RPBE",
    "PW91",
    "BP86",
    "BLYP",
    "B3LYP",
    "PBE0",
    "R2SCAN",
    "TPSS",
    "M06L",
    "TASK",
]

BASIS_SETS = [
    "sto-3g",
    "sto-6g",
    "3-21G",
    "4-31G",
    "6-31G",
    "6-31+G",
    "6-31++G",
    "cc-pvDZ",
    "def2-SVP",
    "def2-TZVP",
]

EXAMPLE_FILES = {
    "Water": "H2O.xyz",
    "Hydrogen": "H2.xyz",
    "Ethane": "Ethane.xyz",
    "Silver chloride": "AgCl.xyz",
    "Cadmium dimer": "Cd_dimer.xyz",
}


def load_example_xyz(example_name):
    path = APP_ROOT / "structures" / EXAMPLE_FILES[example_name]
    return path.read_text(encoding="utf-8").strip()


def parse_xyz(xyz_text):
    return ase_read(io.StringIO(xyz_text), format="xyz")


def structure_view(atoms, style="ball-stick", width=420, height=400):
    xyz_text = atoms_to_xyz(atoms)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(xyz_text, "xyz")
    if style == "ball-stick":
        viewer.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.3}})
    elif style == "ball":
        viewer.setStyle({"sphere": {"scale": 0.4}})
    else:
        viewer.setStyle({"stick": {}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer


def render_structure(atoms, style):
    viewer = structure_view(atoms, style=style)
    try:
        components.html(viewer._make_html(), width=440, height=420)
    except Exception:
        javascript = viewer.js()
        components.html(
            f"{javascript.startjs}{javascript.endjs}",
            width=440,
            height=420,
        )
    st.caption(
        f"{atoms.get_chemical_formula()} · {len(atoms)} atoms · "
        f"{len(set(atoms.get_chemical_symbols()))} element type(s)",
    )


def set_thread_environment(ncores):
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(ncores)


st.title("📐 Geometry Optimization")
st.write(
    "Relax a molecular structure with PyFock DFT energies and forces, using an "
    "optimizer from the Atomic Simulation Environment (ASE)."
)
st.markdown("---")

st.header("1. Molecule Input")
input_column, preview_column = st.columns([1.3, 1])

with input_column:
    molecule_choice = st.selectbox(
        "Select an example molecule or enter a custom XYZ:",
        list(EXAMPLE_FILES) + ["Custom"],
        key="go_molecule_choice",
    )
    if molecule_choice == "Custom":
        xyz_content = st.text_area(
            "Paste XYZ coordinates:",
            height=230,
            placeholder=(
                "3\nWater molecule\nO 0.0 0.0 0.11779\n"
                "H 0.0 0.75545 -0.47116\nH 0.0 -0.75545 -0.47116"
            ),
            key="go_custom_xyz",
        )
    else:
        xyz_content = st.text_area(
            "XYZ coordinates:",
            value=load_example_xyz(molecule_choice),
            height=230,
            key=f"go_xyz_{molecule_choice}",
        )

parsed_atoms = None
parse_error = None
if xyz_content and xyz_content.strip():
    try:
        parsed_atoms = parse_xyz(xyz_content)
    except Exception as exc:
        parse_error = str(exc)

with preview_column:
    st.subheader("Molecule Preview")
    visualization_style = st.selectbox(
        "Visualization style:",
        ["ball-stick", "stick", "ball"],
        key="go_visualization_style",
    )
    if parsed_atoms is not None:
        render_structure(parsed_atoms, visualization_style)
    elif parse_error:
        st.error(f"Could not read the XYZ coordinates: {parse_error}")
    else:
        st.info("Enter XYZ coordinates to preview the molecule.")

st.markdown("---")
st.header("2. DFT and Optimization Settings")
dft_column, optimization_column = st.columns(2)

with dft_column:
    st.subheader("PyFock DFT")
    basis_set = st.selectbox("Basis Set:", BASIS_SETS, index=0, key="go_basis")
    auxiliary_basis = st.text_input(
        "Auxiliary Basis:",
        value="def2-universal-jfit",
        key="go_auxbasis",
    )
    xc_functional = st.selectbox(
        "XC Functional:",
        XC_FUNCTIONALS,
        index=0,
        key="go_functional",
    )
    scf_max_iterations = st.number_input(
        "Maximum SCF Iterations:",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
        key="go_scf_iterations",
    )
    scf_convergence = st.number_input(
        "SCF Convergence Criterion:",
        min_value=1.0e-8,
        max_value=1.0e-3,
        value=1.0e-6,
        format="%.1e",
        key="go_scf_convergence",
    )
    ao_basis_type = st.selectbox(
        "AO Basis Type:",
        ["CAO", "SAO"],
        index=1,
        key="go_ao_basis_type",
    )
    use_sao_basis = ao_basis_type == "SAO"
    grid_level = st.number_input(
        "Grid Level:", min_value=0, max_value=5, value=3, step=1,
        disabled=xc_functional == "HF", key="go_grid_level",
        help="PyFock grid accuracy: 0 is coarsest and 5 is finest. Not used for HF.",
    )
    initial_guess = st.selectbox(
        "Initial Guess:", ["sano", "core"], key="go_initial_guess",
        format_func=lambda value: "SANO" if value == "sano" else "Core",
        help="PyFock builds the starting density from atomic natural orbitals (SANO) or the core Hamiltonian.",
    )
    if xc_functional == "HF":
        st.warning(
            "PyFock automatically falls back to numerical forces for HF, so each "
            "optimization cycle can require many additional SCF calculations."
        )

with optimization_column:
    st.subheader("ASE Geometry Optimization")
    optimizer_name = st.selectbox(
        "Optimizer:",
        list(OPTIMIZERS),
        index=list(OPTIMIZERS).index("LBFGS"),
        key="go_optimizer",
    )
    st.caption(OPTIMIZER_HELP[optimizer_name])
    force_convergence = st.number_input(
        "Maximum Force Convergence (eV/Å):",
        min_value=0.001,
        max_value=0.100,
        value=0.020,
        step=0.001,
        format="%.3f",
        key="go_force_convergence",
        help="The optimization converges when every atomic force norm is below this value.",
    )
    max_cycles = st.number_input(
        "Maximum Optimization Cycles:",
        min_value=1,
        max_value=MAX_OPTIMIZATION_CYCLES,
        value=MAX_OPTIMIZATION_CYCLES,
        step=1,
        key="go_max_cycles",
    )
    st.info(
        f"The hosted app is limited to **{MAX_OPTIMIZATION_CYCLES} optimization "
        "cycles** and **120 basis functions**. "
        f"[Download PyFock GUI from GitHub]({GUI_GITHUB_URL}) and run it locally "
        "for longer optimizations, larger basis sets, and more compute resources."
    )
    st.caption(
        "The initial geometry is frame 0, so the trajectory can contain one more "
        "frame than the selected number of optimization cycles."
    )

st.markdown("---")
run_clicked = st.button(
    "🚀 Optimize Geometry",
    type="primary",
    use_container_width=True,
)

calculation_results = None
if run_clicked:
    if not xyz_content or not xyz_content.strip():
        st.error("Please provide XYZ coordinates.")
        st.stop()
    if parsed_atoms is None:
        st.error(f"Please correct the XYZ input before running: {parse_error}")
        st.stop()
    if not auxiliary_basis.strip() and xc_functional != "HF":
        st.error("Please provide an auxiliary basis for a density-fitted DFT calculation.")
        st.stop()

    electron_count = int(np.sum(parsed_atoms.numbers))
    if electron_count % 2:
        st.error(
            "This PyFock workflow currently supports neutral, closed-shell systems "
            "only. The supplied molecule has an odd number of electrons."
        )
        st.stop()

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    table_placeholder = st.empty()
    status_placeholder.info("Loading PyFock and validating the basis set...")

    ncores = 1
    set_thread_environment(ncores)
    started_at = time.perf_counter()

    try:
        from pyfock import Basis, Mol, PyFockCalculator

        with tempfile.TemporaryDirectory(prefix="pyfock_geometry_opt_") as run_dir:
            run_path = Path(run_dir)
            input_path = run_path / "molecule.xyz"
            input_path.write_text(xyz_content, encoding="utf-8")

            mol = Mol(coordfile=str(input_path), charge=0)
            basis_object = Basis(
                mol,
                {"all": Basis.load(mol=mol, basis_name=basis_set)},
            )
            number_of_basis_functions = int(basis_object.bfs_nao)
            if number_of_basis_functions > 120:
                st.error(
                    f"This system has {number_of_basis_functions} basis functions, "
                    "which exceeds the hosted-app limit of 120. Use a smaller molecule "
                    "or basis set, or run the app locally."
                )
                st.stop()

            atoms = parsed_atoms.copy()
            calculator_kwargs = {
                "functional": xc_functional,
                "basis": basis_set,
                "auxbasis": auxiliary_basis.strip() or None,
                "charge": 0,
                "ncores": ncores,
                "DF": xc_functional != "HF",
                "save_ao_values": True,
                "sao": use_sao_basis,
                "conv_crit": float(scf_convergence),
                "max_itr": int(scf_max_iterations),
                "gridsLevel": int(grid_level),
                "dmat_guess_method": initial_guess,
                "directory": str(run_path / "calculation"),
            }
            atoms.calc = PyFockCalculator(**calculator_kwargs)

            optimizer_log = io.StringIO()
            optimizer_class = OPTIMIZERS[optimizer_name]
            optimizer = optimizer_class(atoms, logfile=optimizer_log)
            trajectory_frames = []
            optimization_records = []

            def record_optimization_frame():
                frame, energy, forces = snapshot_with_results(atoms)
                cycle = int(optimizer.get_number_of_steps())
                trajectory_frames.append(frame)
                optimization_records.append(
                    {
                        "Cycle": cycle,
                        "Energy (eV)": energy,
                        "Maximum Force (eV/Å)": maximum_force(forces),
                        "SCF Iterations": atoms.calc.pyfock_results.get("niter"),
                    }
                )
                progress_bar.progress(min(cycle / int(max_cycles), 1.0))
                status_placeholder.info(
                    f"Optimization cycle {cycle}/{int(max_cycles)} · "
                    f"maximum force {maximum_force(forces):.5f} eV/Å"
                )
                table_placeholder.dataframe(
                    pd.DataFrame(optimization_records).style.format(
                        {
                            "Energy (eV)": "{:.8f}",
                            "Maximum Force (eV/Å)": "{:.6f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            optimizer.attach(record_optimization_frame, interval=1)
            converged = bool(
                optimizer.run(
                    fmax=float(force_convergence),
                    steps=int(max_cycles),
                )
            )

            if not trajectory_frames:
                record_optimization_frame()

            final_frame = trajectory_frames[-1]
            trajectory_extxyz = frames_to_extxyz(trajectory_frames)
            optimized_extxyz = frames_to_extxyz([final_frame])
            optimized_xyz = atoms_to_xyz(final_frame)
            reproduction_script = build_reproduction_script(
                xyz_content=xyz_content,
                optimizer_name=optimizer_name,
                force_convergence=float(force_convergence),
                functional=xc_functional,
                basis_set=basis_set,
                auxiliary_basis=auxiliary_basis.strip() or "def2-universal-jfit",
                scf_convergence=float(scf_convergence),
                scf_max_iterations=int(scf_max_iterations),
                use_sao_basis=use_sao_basis,
                grid_level=int(grid_level),
                initial_guess=initial_guess,
                ncores=ncores,
                max_cycles=int(max_cycles),
            )

            pyfock_log_sections = []
            for output_path in sorted(
                (run_path / "calculation").glob("step_*/output_pyfock_singlepoint.txt")
            ):
                pyfock_log_sections.append(
                    f"{output_path.parent.name}\n{output_path.read_text(encoding='utf-8')}"
                )
            pyfock_log = "\n\n".join(pyfock_log_sections)
            if len(pyfock_log) > 100_000:
                pyfock_log = "[Earlier output omitted]\n" + pyfock_log[-100_000:]

            calculation_results = {
                "basis_functions": number_of_basis_functions,
                "converged": converged,
                "elapsed": time.perf_counter() - started_at,
                "final_frame": final_frame,
                "force_method": atoms.calc.pyfock_results.get("force_method_used", "unknown"),
                "optimized_extxyz": optimized_extxyz,
                "optimized_xyz": optimized_xyz,
                "optimizer_log": optimizer_log.getvalue(),
                "records": pd.DataFrame(optimization_records),
                "reproduction_script": reproduction_script,
                "trajectory_extxyz": trajectory_extxyz,
                "pyfock_log": pyfock_log,
            }

        progress_bar.progress(1.0)
        status_placeholder.empty()
        table_placeholder.empty()
    except ImportError as exc:
        progress_bar.empty()
        status_placeholder.empty()
        table_placeholder.empty()
        st.error(f"Could not load a required package: {exc}")
        st.info("Install the project dependencies, including `pyfock` and `ase`, then retry.")
    except Exception as exc:
        progress_bar.empty()
        status_placeholder.empty()
        table_placeholder.empty()
        st.error(f"Geometry optimization failed: {exc}")
        st.info(
            "Check the molecular geometry, basis sets, and SCF settings. "
            "A difficult starting structure may also benefit from FIRE or FIRE2."
        )

if calculation_results is not None:
    records = calculation_results["records"]
    final_record = records.iloc[-1]
    steps_taken = int(final_record["Cycle"])

    st.markdown("---")
    st.header("3. Optimization Results")
    if calculation_results["converged"]:
        st.success(
            f"Geometry optimization converged in {steps_taken} cycle(s)."
        )
    else:
        st.warning(
            f"The geometry did not reach the {force_convergence:.3f} eV/Å force "
            f"criterion within the selected limit of {int(max_cycles)} cycles. "
            f"[Run the app locally]({GUI_GITHUB_URL}) to continue for more cycles."
        )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Final Energy", f"{final_record['Energy (eV)']:.8f} eV")
    metric_columns[1].metric(
        "Final Maximum Force",
        f"{final_record['Maximum Force (eV/Å)']:.6f} eV/Å",
    )
    metric_columns[2].metric("Optimization Cycles", str(steps_taken))
    metric_columns[3].metric("Wall Time", f"{calculation_results['elapsed']:.2f} s")

    st.caption(
        f"{calculation_results['basis_functions']} basis functions · "
        f"{optimizer_name} optimizer · {calculation_results['force_method']} forces"
    )

    st.subheader("Convergence History")
    st.dataframe(
        records.style.format(
            {
                "Energy (eV)": "{:.8f}",
                "Maximum Force (eV/Å)": "{:.6f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=records["Cycle"],
            y=records["Energy (eV)"],
            mode="lines+markers",
            name="Energy",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=records["Cycle"],
            y=records["Maximum Force (eV/Å)"],
            mode="lines+markers",
            name="Maximum force",
        ),
        secondary_y=True,
    )
    figure.add_hline(
        y=float(force_convergence),
        line_dash="dot",
        line_color="green",
        annotation_text="Force convergence target",
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Optimization Cycle", dtick=1)
    figure.update_yaxes(title_text="Energy (eV)", secondary_y=False)
    figure.update_yaxes(title_text="Maximum Force (eV/Å)", secondary_y=True)
    figure.update_layout(
        title="Geometry Optimization Convergence",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Optimized Geometry")
    optimized_view_column, coordinate_column = st.columns([1, 1.2])
    with optimized_view_column:
        render_structure(
            calculation_results["final_frame"],
            visualization_style,
        )
    with coordinate_column:
        final_atoms = calculation_results["final_frame"]
        coordinates = pd.DataFrame(
            {
                "Atom": final_atoms.get_chemical_symbols(),
                "x (Å)": final_atoms.positions[:, 0],
                "y (Å)": final_atoms.positions[:, 1],
                "z (Å)": final_atoms.positions[:, 2],
            }
        )
        st.dataframe(
            coordinates.style.format({"x (Å)": "{:.8f}", "y (Å)": "{:.8f}", "z (Å)": "{:.8f}"}),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Optimized XYZ coordinates"):
            st.code(calculation_results["optimized_xyz"], language="text")

    st.subheader("Downloads")
    st.caption(
        "The extXYZ trajectory stores the energy and three force components for "
        "every atom at every captured optimization frame."
    )

    @st.fragment
    def render_downloads():
        download_columns = st.columns(3)
        with download_columns[0]:
            st.download_button(
                "📥 Optimized Geometry (extXYZ)",
                data=calculation_results["optimized_extxyz"],
                file_name="optimized_geometry.extxyz",
                mime="chemical/x-extxyz",
                use_container_width=True,
            )
        with download_columns[1]:
            st.download_button(
                "📥 Optimization Trajectory (extXYZ)",
                data=calculation_results["trajectory_extxyz"],
                file_name="optimization_trajectory.extxyz",
                mime="chemical/x-extxyz",
                use_container_width=True,
            )
        with download_columns[2]:
            st.download_button(
                "📥 Reproduction Script",
                data=calculation_results["reproduction_script"],
                file_name="pyfock_geometry_optimization.py",
                mime="text/x-python",
                use_container_width=True,
            )

    render_downloads()

    with st.expander("Reproduction script"):
        st.code(calculation_results["reproduction_script"], language="python")
    with st.expander("ASE optimizer log"):
        st.code(calculation_results["optimizer_log"] or "No optimizer log was produced.")
    with st.expander("PyFock calculation log"):
        st.code(calculation_results["pyfock_log"] or "No PyFock log was produced.")
elif not run_clicked:
    st.info(
        "Configure the molecule, DFT method, and ASE optimizer, then click "
        "**Optimize Geometry**."
    )

st.markdown("---")
st.markdown(
    "<div style='text-align: center'><p>PyFock DFT forces · ASE geometry optimization</p></div>",
    unsafe_allow_html=True,
)
