import streamlit as st
import os
import tempfile
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
from io import StringIO
import time
import re
import contextlib
import io as _io
from ase.io import read as ase_read
import pandas as pd
import pyfock
from gui_common import render_sidebar

# Set page configuration
st.set_page_config(
    page_title='PyFock GUI - Interactive DFT Calculations',
    layout='wide',
    page_icon="⚛️",
    menu_items={
        'About': "PyFock GUI - A web interface for PyFock, a pure Python DFT code with Numba JIT acceleration"
    }
)
render_sidebar()

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Helper: create an ASE Atoms object or fallback
def _parse_xyz_to_atoms(xyz_text):
    # ASE can read from string via io
    return ase_read(StringIO(xyz_text), format='xyz')

# get_structure_viz2 implementation (based on sample provided)
def get_structure_viz2(atoms_obj, style='stick', width=400, height=400):
    xyz_str = ""
    xyz_str += f"{len(atoms_obj)}\n"
    xyz_str += "Structure\n"
    for atom in atoms_obj:
        # atom may be ASE Atoms or fallback MiniAtom
        sym = atom.symbol if hasattr(atom, 'symbol') else atom.get_chemical_symbols()[0]
        pos = atom.position if hasattr(atom, 'position') else atom.position
        xyz_str += f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")
    if style.lower() == 'ball-stick':
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
    elif style.lower() == 'stick':
        view.setStyle({'stick': {}})
    elif style.lower() == 'ball':
        view.setStyle({'sphere': {'scale': 0.4}})
    else:
        view.setStyle({'stick': {'radius': 0.15}})
    try:
        pbc_any = atoms_obj.pbc.any()
    except Exception:
        pbc_any = False

    view.zoomTo()
    view.setBackgroundColor('white')
    return view

# === Cube visualization function ===
def visualize_cube_in_component(cube_content, title, iso_val, opac, width=420, height=360):
    # return the HTML for embedding so it can be used inside columns
    view = py3Dmol.view(width=width, height=height)
    view.addModel(cube_content, 'cube')
    view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3},
                    'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})

    # For orbitals, show both lobes; for density, show only positive + negative appropriately
    if 'Density' not in title:
        view.addVolumetricData(cube_content, 'cube',
                                {'isoval': -abs(iso_val), 'color': 'blue', 'opacity': opac})
    view.addVolumetricData(cube_content, 'cube',
                            {'isoval': abs(iso_val), 'color': 'red', 'opacity': opac})

    view.zoomTo()
    view.setClickable({'clickable': 'true'})
    view.enableContextMenu({'contextMenuEnabled': 'true'})
    view.show()
    view.render()
    # don't spin automatically; allow user to toggle via JS later if desired
    t = view.js()
    html_content = f"{t.startjs}{t.endjs}"
    return html_content

# Example XYZ files
EXAMPLE_MOLECULES = {
    "Water": """3
Water molecule
O     0.000000    0.000000    0.117790
H     0.000000    0.755453   -0.471161
H     0.000000   -0.755453   -0.471161""",

    "Acetone": """10
Acetone molecule
O       0.000247197289657     -1.311344859924947      0.000033372829371
C       0.000008761532627     -0.103796835732344      0.000232428229233
C       1.285011287026515      0.689481114475586     -0.000005118102586
C      -1.285310305026895      0.688972899031773     -0.000008308924344
H       1.326033303131908      1.335179621002258     -0.879574382138206
H       1.324106820690737      1.339904097018082      0.876116213728557
H       2.136706543431990      0.014597477886500      0.002551051783708
H      -2.136748467815155      0.013761611436540      0.002550873429523
H      -1.326572603227431      1.334639056170679     -0.879590243114624
H      -1.324682534264540      1.339405821389821      0.876094109485741""",

    "Acetonitrile": """6
Acetonitrile molecule
N       1.238504335855378     -0.000002648443155     -0.000006432209062
C      -1.367114484335133      0.000020833177302      0.000019691866100
C       0.091462639284387      0.000004939432085     -0.000014578846105
H      -1.737603541725548     -0.830506551932429      0.597694569897051
H      -1.737658959168382     -0.102297452169442     -1.018057029309221
H      -1.737589980824850      0.932880879510178      0.420463777423243""",

    "Methane": """5
Methane molecule
C     0.000000    0.000000    0.000000
H     0.629118    0.629118    0.629118
H    -0.629118   -0.629118    0.629118
H    -0.629118    0.629118   -0.629118
H     0.629118   -0.629118   -0.629118""",

    "Benzene": """12
Benzene molecule
C     1.395890    0.000000    0.000000
C     0.697945    1.209021    0.000000
C    -0.697945    1.209021    0.000000
C    -1.395890    0.000000    0.000000
C    -0.697945   -1.209021    0.000000
C     0.697945   -1.209021    0.000000
H     2.482610    0.000000    0.000000
H     1.241305    2.149540    0.000000
H    -1.241305    2.149540    0.000000
H    -2.482610    0.000000    0.000000
H    -1.241305   -2.149540    0.000000
H     1.241305   -2.149540    0.000000""",

    "Ammonia": """4
Ammonia molecule
N     0.000000    0.000000    0.100000
H     0.945000    0.000000   -0.266000
H    -0.472500    0.818000   -0.266000
H    -0.472500   -0.818000   -0.266000""",

    "Carbon Dioxide": """3
Carbon dioxide molecule
C     0.000000    0.000000    0.000000
O     0.000000    0.000000    1.160000
O     0.000000    0.000000   -1.160000""",

    "Hydrogen Peroxide": """4
Hydrogen peroxide molecule
O     0.000000    0.000000    0.000000
O     1.450000    0.000000    0.000000
H     0.000000    0.930000    0.000000
H     1.450000   -0.930000    0.000000""",

    "Formaldehyde": """4
Formaldehyde molecule
C     0.000000    0.000000    0.000000
O     1.200000    0.000000    0.000000
H    -0.550000    0.940000    0.000000
H    -0.550000   -0.940000    0.000000""",

    "Hydrogen Cyanide": """3
Hydrogen cyanide molecule
H     0.000000    0.000000    0.000000
C     1.065000    0.000000    0.000000
N     2.232000    0.000000    0.000000""",

    "Acetylene": """4
Acetylene molecule
H     0.000000    0.000000    0.000000
C     0.601000    0.000000    0.000000
C     1.764000    0.000000    0.000000
H     2.365000    0.000000    0.000000""",

    "Ethylene": """6
Ethylene molecule
C     0.000000    0.000000    0.000000
C     1.339000    0.000000    0.000000
H    -0.540000    0.930000    0.000000
H    -0.540000   -0.930000    0.000000
H     1.879000    0.930000    0.000000
H     1.879000   -0.930000    0.000000""",

    "Ethane": """8
Ethane molecule
C     0.000000    0.000000    0.000000
C     1.540000    0.000000    0.000000
H    -0.540000    0.930000    0.000000
H    -0.540000   -0.930000    0.000000
H     0.000000    0.000000    1.090000
H     2.080000    0.930000    0.000000
H     2.080000   -0.930000    0.000000
H     1.540000    0.000000   -1.090000""",

    "Formic Acid": """5
Formic acid molecule
C     0.000000    0.000000    0.000000
O     1.200000    0.000000    0.000000
O    -0.600000    1.100000    0.000000
H     1.700000    0.900000    0.000000
H    -0.600000   -0.900000    0.000000""",

    "Hydrogen Sulfide": """3
Hydrogen sulfide molecule
S     0.000000    0.000000    0.000000
H     0.960000    0.000000    0.000000
H    -0.480000    0.830000    0.000000""",

    "Tetrahydrofuran": """13
Tetrahydrofuran molecule
O       1.216699382773870     -0.000516422279060     -0.000000629115220
C      -1.016993686921708     -0.728737145702576     -0.227053769599122
C      -1.016379073694511      0.729554513122363      0.227039749920194
C       0.395888395971846     -1.160306071902647      0.143905791563541
C       0.396849106350064      1.159943366322044     -0.143953936744947
H      -1.782338327214497     -1.336082444384147      0.254184465508221
H      -1.159922021048819     -0.787714532025205     -1.307880287340100
H      -1.781241112969484      1.337526054547069     -0.254174631443963
H      -1.159223957468540      0.788641683930875      1.307872257115590
H       0.441717113009142     -1.507461993885669      1.181993316376597
H       0.789985254249466     -1.947807118924544     -0.499943162908684
H       0.442962196586526      1.507022288075474     -1.182056728365057
H       0.791596740442480      1.947137848999537      0.499867568609654""",

    "Pyrrole": """10
Pyrrole molecule
N       0.003181105319591     -1.154989666506124      0.000060405177869
C      -1.117737448486888     -0.370847266235924     -0.000011684111207
C       1.119766547077291     -0.364686948222411      0.000018496752116
C      -0.713366217362578      0.937235325591941     -0.000096441964893
C       0.708206650120975      0.941149612298515     -0.000053474487459
H       0.005944376523274     -2.157707636293536     -0.000153625688957
H      -2.102541509509273     -0.805490708525106      0.000110657494436
H       2.106949265704444     -0.793899550618045      0.000097097601888
H      -1.362506178052089      1.796728992178547     -0.000156325233850
H       1.352603398906602      1.804207848609518     -0.000015105544776""",

    "Dimethyl Ether": """9
Dimethyl ether molecule
O       0.000004977280923      0.530020309529034     -0.000005108317779
C       1.164212882530410     -0.260593898872961      0.000001269096639
C      -1.164190191714442     -0.260612616856000     -0.000020896710708
H       1.210503469986643     -0.900685141083738      0.889522583088480
H       2.020207692369726      0.410753932008613      0.000001864695406
H       1.210505099946668     -0.900685096089216     -0.889519959770612
H      -1.210516092616227     -0.900663538561855      0.889526642712922
H      -1.210427870830911     -0.900750324054947     -0.889509722206098
H      -2.020199970977986      0.410716373006091     -0.000096671211569""",

    "AgCl": """2
AgCl molecule
Ag    0.000000    0.000000    0.000000
Cl    0.000000    0.000000    2.280000""",

    "AuCl": """2
AuCl molecule
Au    0.000000    0.000000    0.000000
Cl    0.000000    0.000000    2.230000""",

    "Cd dimer": """2
Cd dimer
Cd    0.000000    0.000000    0.000000
Cd    0.000000    0.000000    2.980000""",

}

# Natively implemented PyFock functionals
XC_FUNCTIONALS = {
    "LDA": "LDA",
    "HF": "HF",
    "PBE": "PBE",
    "PBESOL": "PBESOL",
    "RPBE": "RPBE",
    "PW91": "PW91",
    "BP86": "BP86",
    "BLYP": "BLYP",
    "B3LYP": "B3LYP",
    "PBE0": "PBE0",
    "R2SCAN": "R2SCAN",
    "TPSS": "TPSS",
    "M06L": "M06L",
    "TASK": "TASK",
}
PYSCF_XC_FUNCTIONALS = {
    "LDA": "1,7",
    "SVWN5": "1,7",
    "SPZ": "1,9",
    "SPZMOD": "1,10",
    "SPW": "1,12",
    "SPWMOD": "1,12",
    "PBE": "101,130",
    "PBESOL": "116,133",
    "RPBE": "117,130",
    "PW91": "109,134",
    "BP86": "106,132",
    "BLYP": "106,131",
    "B3LYP": "B3LYP",
    "PBE0": "PBE0",
    "R2SCAN": "497,498",
    "TPSS": "202,231",
    "M06L": "203,233",
    "TASK": "707,7",
}

BASIS_SETS = ["sto-3g", "sto-6g", "3-21G", "4-31G", "6-31G", "6-31+G", "6-31++G", "cc-pvDZ", "def2-SVP", "def2-TZVP"]

# Main title
st.title("⚛️ PyFock GUI - Interactive DFT Calculations")
st.markdown("---")

# Input section
st.header("1. DFT Setup")

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("Molecule Input")

    # Molecule selection
    molecule_choice = st.selectbox(
        "Select example molecule or paste custom XYZ:",
        [
            "Water",
            "Acetone",
            "Tetrahydrofuran",
            "Pyrrole",
            "Dimethyl Ether",
            "Benzene",
            "Carbon Dioxide",
            "Hydrogen Peroxide",
            "Formic Acid",
            "Hydrogen Sulfide",
            "Methane",
            "AgCl",
            # "AuCl",
            "Cd dimer",
            "Custom"
        ]
    )

    if molecule_choice == "Custom":
        xyz_content = st.text_area(
            "Paste XYZ coordinates:",
            height=200,
            placeholder="3\nWater molecule\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0"
        )
    else:
        xyz_content = st.text_area(
            "XYZ coordinates:",
            value=EXAMPLE_MOLECULES[molecule_choice],
            height=200
        )

    # === NEW: Structure visualization right at molecule selection ===
    # This uses ASE if available; otherwise, show a simple py3Dmol view from the XYZ string.
    with col2:
        if xyz_content and xyz_content.strip():
            st.markdown("### Molecule Visualization", unsafe_allow_html=True)
            viz_style = st.selectbox("Select Visualization Style:", ["ball-stick", "stick", "ball"], key="viz_style_select")
            atoms_obj = _parse_xyz_to_atoms(xyz_content)

            # Render py3Dmol
            view_3d = get_structure_viz2(atoms_obj, style=viz_style, width=400, height=400)
            # Use components.html to insert the viewer HTML
            try:
                st.components.v1.html(view_3d._make_html(), width=420, height=420)
            except Exception:
                # fallback to js html
                t = view_3d.js()
                html_content = f"{t.startjs}{t.endjs}"
                components.html(html_content, height=420, width=420)

            # Structure information
            st.markdown("### Structure Information")
            atoms_info = {
                "Number of Atoms": len(atoms_obj),
                "Chemical Formula": atoms_obj.get_chemical_formula() if hasattr(atoms_obj, 'get_chemical_formula') else "".join(atoms_obj.get_chemical_symbols()),
                "Atom Types": ", ".join(sorted(list(set(atoms_obj.get_chemical_symbols()))))
            }

            for key, value in atoms_info.items():
                st.write(f"**{key}:** {value}")

with col1:
    st.subheader("Calculation Settings")

    basis_set = st.selectbox("Basis Set:", BASIS_SETS, index=0)
    auxbasis = st.text_input("Auxiliary Basis:", value="def2-universal-jfit")

    xc_functional = st.selectbox(
        "XC Functional:",
        list(XC_FUNCTIONALS.keys()),
        index=0
    )

    default_max_iterations = 20
    default_conv_crit = 1e-6
    max_iterations = st.number_input("Max Iterations:", min_value=1, max_value=50, value=default_max_iterations)
    conv_crit = st.number_input("Convergence Criterion:", min_value=1e-8, max_value=1e-3, value=default_conv_crit, format="%.1e")
    ao_basis_type = st.selectbox("AO Basis Type:", ["CAO", "SAO"], index=1)
    use_sao_basis = ao_basis_type == "SAO"
    ncores = 1#st.number_input("Number of Cores:", min_value=1, max_value=8, value=4)
    grid_level = st.number_input(
        "Grid Level:", min_value=0, max_value=5, value=3, step=1,
        disabled=xc_functional == "HF",
        help="PyFock grid accuracy: 0 is coarsest and 5 is finest. Not used for HF.",
    )
    initial_guess = st.selectbox(
        "Initial Guess:", ["sano", "core"],
        format_func=lambda value: "SANO" if value == "sano" else "Core",
        help="PyFock builds the starting density from atomic natural orbitals (SANO) or the core Hamiltonian.",
    )

st.markdown("---")

from calculation_workflow import (
    run_scf, calculate_forces, calculate_dipole, generate_cube,
    compare_energy, build_input_script,
)

settings = dict(
    xyz_content=xyz_content, basis_set=basis_set, auxbasis=auxbasis,
    xc_functional=xc_functional, grid_level=int(grid_level),
    initial_guess=initial_guess, conv_crit=float(conv_crit),
    max_iterations=int(max_iterations), ncores=ncores, use_sao_basis=use_sao_basis,
)

# Invalidate all calculation output when an SCF input changes. Display-only
# controls are outside this snapshot and can still reuse the saved result.
previous_settings = st.session_state.get(
    'scf_input_settings', st.session_state.get('scf_result', {}).get('settings')
)
if previous_settings is not None and previous_settings != settings:
    st.session_state.pop('scf_result', None)
    st.session_state.pop('scf_error', None)
st.session_state.scf_input_settings = dict(settings)

if st.button("🚀 Run DFT Calculation", type="primary"):
    log = _io.StringIO()
    try:
        if not xyz_content.strip():
            raise ValueError("Please provide XYZ coordinates.")
        with st.spinner("Running SCF calculation..."):
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                completed = run_scf(settings)
        completed['log'] = log.getvalue()
        st.session_state.scf_result = completed
        st.session_state.pop('scf_error', None)
    except Exception as exc:
        st.session_state.scf_error = (str(exc), log.getvalue())

if 'scf_error' in st.session_state:
    error, error_log = st.session_state.scf_error
    st.error(f"Calculation failed: {error}")
    with st.expander("Failed calculation output"):
        st.code(strip_ansi(error_log))

result = st.session_state.get('scf_result')
if result is None:
    st.info("Choose your settings and run DFT. Forces, dipole moments, and orbital or density plots become available after SCF.")
else:
    dftObj = result['dft']
    dmat = result['dmat']
    energyPyFock = result['energy']
    saved = result['settings']
    st.caption(f"Completed calculation: {saved['xc_functional']} · {saved['basis_set']} · {'SAO' if saved['use_sao_basis'] else 'CAO'} · grid {saved['grid_level']} · {saved['initial_guess'].upper()} guess")
    st.header("2. SCF Results")
    if dftObj.converged:
        st.success(f"SCF converged in {dftObj.niter} iterations ({result['elapsed']:.2f} seconds).")
    else:
        st.warning(f"SCF did not converge in {dftObj.niter} iterations. Properties from this density may be inaccurate; forces require convergence.")
    # Energy and basic properties
    col6, col7, col8 = st.columns(3)

    with col6:
        st.metric("Total Energy (PyFock)", f"{energyPyFock:.8f} Ha")

        with st.expander("Energy Components"):
            import pandas as pd
            energy_df = pd.DataFrame({
                "Component": [
                    "Kinetic Energy",
                    "Nuclear-Electron Attraction",
                    "Electron-Electron Repulsion",
                    "Exchange-Correlation",
                    "Nuclear Repulsion",
                    "Exact Exchange",
                    "ECP"
                ],
                "Energy (Ha)": [
                    f"{dftObj.Kinetic_energy:.8f}",
                    f"{dftObj.Nuc_energy:.8f}",
                    f"{dftObj.J_energy:.8f}",
                    "N/A" if dftObj.XC_energy is None else f"{dftObj.XC_energy:.8f}",
                    f"{dftObj.Nuclear_repulsion_energy:.8f}",
                    f"{getattr(dftObj, 'Exx_energy', 0.0):.8f}",
                    f"{getattr(dftObj, 'ECP_energy', 0.0):.8f}"
                ]
            })
            st.dataframe(energy_df, hide_index=True, use_container_width=True)

    with col7:
        # Calculate HOMO-LUMO gap
        occupied = np.where(dftObj.mo_occupations > 1e-8)[0]
        if len(occupied) > 0 and len(occupied) < len(dftObj.mo_energies):
            homo_idx = occupied[-1]
            lumo_idx = homo_idx + 1
            homo_energy = dftObj.mo_energies[homo_idx]
            lumo_energy = dftObj.mo_energies[lumo_idx]
            gap = (lumo_energy - homo_energy) * 27.2114  # Convert to eV
            st.metric("HOMO-LUMO Gap", f"{gap:.4f} eV")
        else:
            homo_idx = None
            lumo_idx = None
            st.metric("HOMO-LUMO Gap", "N/A")

    with col8:
        st.metric("SCF Iterations", f"{dftObj.niter}")

        with st.expander("SCF Convergence Details"):
            # Create a DataFrame for the energies
            import pandas as pd
            scf_data = pd.DataFrame({
                'Iteration': range(1, len(dftObj.scf_energies) + 1),
                'Energy (Ha)': dftObj.scf_energies
            })

            # Calculate energy change between iterations
            scf_data['ΔE (Ha)'] = scf_data['Energy (Ha)'].diff()

            # Display the table
            st.dataframe(scf_data, use_container_width=True, hide_index=True)

            # Plot convergence
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scf_data['Iteration'],
                y=scf_data['Energy (Ha)'],
                mode='lines+markers',
                name='Energy',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title='SCF Energy Convergence',
                xaxis_title='Iteration',
                yaxis_title='Energy (Hartree)',
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # MO energies
    st.subheader("Molecular Orbital Energies")
    mo_energies_ev = dftObj.mo_energies * 27.2114  # Convert to eV

    col9, col10 = st.columns(2)
    with col9:
        if homo_idx is not None:
            st.write(f"**HOMO (orbital {homo_idx}):** {mo_energies_ev[homo_idx]:.4f} eV")
    with col10:
        if lumo_idx is not None:
            st.write(f"**LUMO (orbital {lumo_idx}):** {mo_energies_ev[lumo_idx]:.4f} eV")

    # Show MO energies
    with st.expander("View All MO Energies"):
        mo_data = {
            "Orbital": list(range(len(mo_energies_ev))),
            "Energy (eV)": [f"{e:.6f}" for e in mo_energies_ev],
            "Occupation": dftObj.mo_occupations
        }
        st.dataframe(mo_data, height=300)
    # Density matrix expander and download ===
    with st.expander("Density Matrix (dmat) — view / download"):
        try:
            st.write(dmat)

        except Exception as e:
            st.write("Failed to show density matrix:", str(e))


    st.download_button("Download density matrix", pd.DataFrame(dmat).to_csv(index=False), "density_matrix.csv", "text/csv")
    script_container = st.container()
    output_container = st.container()

    st.header("3. Optional Calculations")
    st.caption("These actions use the saved SCF result. Changing plot settings does not repeat SCF or generate cubes.")

    def perform_action(label, action, compute, save):
        log = _io.StringIO()
        succeeded = False
        try:
            with st.spinner(label):
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    value = compute()
            save(value)
            if action not in result['actions']:
                result['actions'].append(action)
            succeeded = True
        except Exception as exc:
            st.error(f"{label} failed: {exc}")
        finally:
            result['log'] += f"\n--- {label} ---\n" + log.getvalue()
        if succeeded and action['kind'] in ('forces', 'dipole', 'comparison'):
            # Refresh buttons rendered before the result was saved. SCF and
            # completed properties are retained and are not recalculated.
            st.rerun()

    force_col, dipole_col, reference_col = st.columns(3)
    with force_col:
        force_supported = dftObj.converged and dftObj.xc != 'HF' and getattr(dftObj, 'exx_coef', 0.0) == 0 and dftObj.isDF
        if st.button("Calculate forces", disabled=not force_supported or result['forces'] is not None):
            perform_action("Calculating forces", {'kind': 'forces'},
                           lambda: calculate_forces(result), lambda value: result.update(forces=value))
        if not force_supported:
            st.caption("PyFock analytical forces require converged, density-fitted, pure DFT. HF and hybrids currently require numerical forces with additional SCF runs.")
    with dipole_col:
        if st.button("Calculate dipole moment", disabled=result['dipole'] is not None):
            perform_action("Calculating dipole moment", {'kind': 'dipole'},
                           lambda: calculate_dipole(result), lambda value: result.update(dipole=value))
    with reference_col:
        if st.button("Compare energy with PySCF", disabled=result['comparison'] is not None):
            perform_action("PySCF comparison", {'kind': 'comparison', 'xc': PYSCF_XC_FUNCTIONALS.get(saved['xc_functional'], 'HF')},
                           lambda: compare_energy(result, PYSCF_XC_FUNCTIONALS), lambda value: result.update(comparison=value))

    if result['forces'] is not None:
        st.subheader("Atomic Forces")
        forces = pd.DataFrame(result['forces'], columns=['Fx (Ha/Bohr)', 'Fy (Ha/Bohr)', 'Fz (Ha/Bohr)'])
        forces.insert(0, 'Atom', _parse_xyz_to_atoms(saved['xyz_content']).get_chemical_symbols())
        st.dataframe(forces, hide_index=True)
        st.download_button("Download forces", forces.to_csv(index=False), "forces.csv", "text/csv")
        atoms = _parse_xyz_to_atoms(saved['xyz_content'])
        force_vectors = np.asarray(result['forces'])
        force_norms = np.linalg.norm(force_vectors, axis=1)
        maximum_force = float(force_norms.max())
        view = get_structure_viz2(atoms, style='ball-stick', width=700, height=450)
        if maximum_force > 1e-10:
            arrow_scale = st.slider("Force arrow scale", 0.2, 2.0, 1.0, 0.1)
            molecule_size = max(float(np.ptp(atoms.positions, axis=0).max()), 1.0)
            display_scale = 0.65 * molecule_size * arrow_scale / maximum_force
            for index, (position, force, norm) in enumerate(zip(atoms.positions, force_vectors, force_norms)):
                if norm <= 1e-10:
                    continue
                end = position + force * display_scale
                view.addArrow({
                    'start': dict(zip(('x', 'y', 'z'), position.tolist())),
                    'end': dict(zip(('x', 'y', 'z'), end.tolist())),
                    'radius': 0.06, 'radiusRatio': 2.5, 'mid': 0.75,
                    'color': '#e45756',
                })
                view.addLabel(f'F{index + 1}', {
                    'position': dict(zip(('x', 'y', 'z'), end.tolist())),
                    'fontColor': '#e45756', 'backgroundOpacity': 0,
                    'fontSize': 16,
                })
            view.zoomTo()
            view.zoom(0.8)
        components.html(view._make_html(), height=470)
        if maximum_force > 1e-10:
            st.caption("Each arrow starts at its atom and points along its force. All arrows share one display scale, so their relative lengths reflect force magnitudes. F1, F2, … follow the atom order above. Drag to rotate; adjusting the scale does not recalculate forces.")
        else:
            st.caption("All forces are effectively zero; no direction arrows are shown.")
        with st.expander("How forces are calculated", expanded=True):
            st.latex(r"\mathbf{F}_A = -\nabla_{\mathbf{R}_A} E_{\mathrm{tot}}, \qquad F_{A\alpha} = -\frac{\partial E_{\mathrm{tot}}}{\partial R_{A\alpha}},\quad \alpha\in\{x,y,z\}")
            st.markdown("The force on atom **A** is the negative derivative of the total energy with respect to its position **Rₐ**. It points toward decreasing energy. PyFock evaluates analytical derivatives from the saved, converged SCF result, including the basis-motion (Pulay) terms. Forces are reported in **Hartree/Bohr**.")
    if result['dipole'] is not None:
        dipole = result['dipole'] * 2.541746473
        st.subheader("Dipole Moment")
        st.metric("Total dipole", f"{np.linalg.norm(dipole):.5f} Debye")
        st.dataframe(pd.DataFrame({'Component': ['X', 'Y', 'Z'], 'Dipole (Debye)': dipole}), hide_index=True)
        atoms = _parse_xyz_to_atoms(saved['xyz_content'])
        positions = atoms.positions
        center = positions.mean(axis=0)
        magnitude = np.linalg.norm(dipole)
        length = max(float(np.ptp(positions, axis=0).max()), 1.0)
        vector = dipole / magnitude * length if magnitude > 1e-10 else np.zeros(3)
        end = center + vector
        view = get_structure_viz2(atoms, style='ball-stick', width=700, height=450)
        if magnitude > 1e-10:
            view.addArrow({
                'start': dict(zip(('x', 'y', 'z'), center.tolist())),
                'end': dict(zip(('x', 'y', 'z'), end.tolist())),
                'radius': 0.08, 'radiusRatio': 2.5, 'mid': 0.75,
                'color': '#e45756',
            })
            view.addLabel('μ', {
                'position': dict(zip(('x', 'y', 'z'), end.tolist())),
                'fontColor': '#e45756', 'backgroundOpacity': 0,
                'fontSize': 24,
            })
            view.zoomTo()
            view.zoom(0.8)
        components.html(view._make_html(), height=470)
        if magnitude > 1e-10:
            st.caption("The dipole vector points from negative toward positive charge. Arrow length is scaled for visibility. Drag to rotate the molecule and vector together.")
        else:
            st.caption("The dipole is effectively zero, so it has no defined direction.")
        with st.expander("How the dipole moment is calculated", expanded=True):
            st.latex(r"\boldsymbol{\mu} = \sum_A Z_A\mathbf{R}_A - \int \mathbf{r}\,\rho(\mathbf{r})\,d^3r")
            st.latex(r"\mu_\alpha = \sum_A Z_A R_{A\alpha} - \sum_{ij} D_{ji}\langle\chi_i|r_\alpha|\chi_j\rangle, \qquad \alpha\in\{x,y,z\}")
            st.markdown("In atomic units, the dipole is the nuclear contribution minus the electronic contribution. **Zₐ** is the nuclear charge (the effective ionic charge when using an ECP), **Rₐ** is the nuclear position, and **ρ** is the electron number density. **D** is the saved density matrix and **χᵢ** are the atomic basis functions; PyFock evaluates their position integrals without rerunning SCF. Coordinates are measured from the XYZ origin, in Bohr.")
            st.latex(r"1\ e a_0 = 2.541746473\ \mathrm{Debye}, \qquad |\boldsymbol{\mu}|=\sqrt{\mu_x^2+\mu_y^2+\mu_z^2}")
    if result['comparison'] is not None:
        comparison = result['comparison']
        st.subheader("PySCF Comparison")
        st.metric("PySCF energy", f"{comparison['energy']:.8f} Ha")
        st.metric("Absolute energy difference", f"{abs(comparison['energy'] - result['energy'])*1000:.6f} mHa")
        if not comparison['converged']:
            st.warning("The PySCF reference did not converge.")

    with st.expander("MO and Density Plots", expanded=bool(result['cubes'])):
        cube_resolution = st.slider("Cube File Resolution (nx=ny=nz):", 30, 50, 40)
        isovalue = st.number_input("Isovalue:", 0.001, 1.0, value=0.05, step=0.001, format="%.6f")
        opacity = st.slider("Opacity:", 0.0, 1.0, value=0.90, step=0.01)
        orbital = st.number_input("Orbital index (0-based):", min_value=0, max_value=len(dftObj.mo_energies)-1,
                                  value=int(homo_idx) if homo_idx is not None else 0, step=1)
        requests = []
        buttons = st.columns(3)
        if buttons[0].button("Plot selected MO"):
            requests.append(int(orbital))
        if buttons[1].button("Plot HOMO and LUMO", disabled=homo_idx is None):
            requests.extend([int(homo_idx), int(lumo_idx)])
        if buttons[2].button("Plot electron density"):
            requests.append(None)
        for index in requests:
            key = (index, cube_resolution)
            if key not in result['cubes']:
                action = {'kind': 'cube', 'orbital': index, 'resolution': cube_resolution}
                perform_action("Generating cube", action, lambda: generate_cube(result, index, cube_resolution),
                               lambda value: result['cubes'].update({key: value}))
        for (index, resolution), content in result['cubes'].items():
            title = 'Density' if index is None else f'MO {index}'
            st.markdown(f"#### {title} · {resolution}³ points")
            components.html(visualize_cube_in_component(content, title, isovalue, opacity), height=380, width=420)
            filename = f"density_{resolution}.cube" if index is None else f"MO_{index}_{resolution}.cube"
            st.download_button(f"Download {title} ({resolution}³)", content, filename, key=f'cube_{index}_{resolution}')

    python_script = build_input_script(result)
    with script_container:
        with st.expander("Input Script", expanded=True):
            st.code(python_script, language='python')
        st.download_button("Download input script", python_script, "pyfock_calculation.py", "text/x-python")
    with output_container:
        with st.expander("Calculation Output"):
            st.code(strip_ansi(result['log']))
        st.download_button("Download output text", strip_ansi(result['log']), "pyfock_output.txt", "text/plain")
    if result['actions']:
        with st.expander("Updated Input Script — includes requested calculations"):
            st.code(python_script, language='python')
        st.download_button("Download updated input script", python_script, "pyfock_calculation.py", "text/x-python", key='updated_input_download')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>PyFock GUI - Pure Python DFT with Numba JIT acceleration</p>
    <p>⚡ Fast • 🎯 Accurate • 🐍 Pure Python</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.write('PyFock version being used for this GUI:', pyfock.__version__)
