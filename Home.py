import streamlit as st
import os
import tempfile
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
from io import StringIO
import time
from pyscf import gto, dft, scf
from pyscf.dft import gen_grid
import re
import base64
import contextlib
import sys
import io as _io
from ase import Atoms
from ase.io import read as ase_read
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title='PyFock GUI - Interactive DFT Calculations',
    layout='wide',
    page_icon="⚛️",
    menu_items={
        'About': "PyFock GUI - A web interface for PyFock, a pure Python DFT code with Numba JIT acceleration"
    }
)
# === Background video styling ===
# CSS for Background Video and Content Styling
def set_css():
    st.markdown("""
        <style>
            #myVideo {
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%; 
                min-height: 100%;
                opacity: 0.12; /* <--- adjust video opacity here (0 to 1) */
                pointer-events: none;  /* FIX: allow scrolling */
            }
            .content {
                position: fixed;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                color: #f1f1f1;
                width: 100%;
                padding: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

# Embed Background Video
def embed_video():
    video_link = "https://raw.githubusercontent.com/manassharma07/Website_Files_for_PyFock/main/background_video_pyfock.mp4"
    st.sidebar.markdown(f"""
        <video autoplay muted loop id="myVideo">
            <source src="{video_link}">
            Your browser does not support HTML5 video.
        </video>
    """, unsafe_allow_html=True)

set_css()
embed_video()

# Set page configuration
# st.set_page_config(
#     page_title='PyFock GUI - Interactive DFT Calculations',
#     layout='wide',
#     page_icon="⚛️",
#     menu_items={
#         'About': "PyFock GUI - A web interface for PyFock, a pure Python DFT code with Numba JIT acceleration"
#     }
# )

# Sidebar with enhanced styling
st.sidebar.image("https://raw.githubusercontent.com/manassharma07/PyFock/main/logo_crysx_pyfock.png", use_container_width=True)

st.sidebar.markdown("---")

# About PyFock section
st.sidebar.markdown("### About PyFock")
st.sidebar.markdown("""
**Pure Python DFT** with performance matching C++ codes!

**Key Advantages:**
- 100% Pure Python (including molecular integrals)
- Numba JIT acceleration
- GPU support (CUDA via CuPy)
- Near-quadratic scaling (~O(N²·⁰⁵))
- Accuracy matching PySCF (<10⁻⁷ Ha)
- Windows/Linux/MacOS compatible
- Easy pip installation
""")

st.sidebar.markdown("---")

# Features
st.sidebar.markdown("### GUI Features")
st.sidebar.markdown("""
* Run DFT in your browser
* Visualize HOMO, LUMO, density
* Compare with PySCF
* Download cube files & scripts
* Interactive 3D visualization
* No installation required!
""")

st.sidebar.markdown("---")

# Links section
st.sidebar.markdown("### 🔗 Links & Resources")
st.sidebar.markdown("""
[![GitHub (PyFock)](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/manassharma07/PyFock)
[![GitHub (PyFock GUI)](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/manassharma07/PyFock-GUI)
[![PyPI](https://img.shields.io/badge/PyPI-Package-orange?logo=pypi)](https://pypi.org/project/pyfock/)
[![Docs](https://img.shields.io/badge/Documentation-Read-green?logo=readthedocs)](https://pyfock-docs.bragitoff.com)

📄 **Article:** *(coming soon)*

👨‍💻 **Developer:** [Manas Sharma](https://www.linkedin.com/in/manassharma07)

⭐ **Star the repo** if you find it useful!
""")

st.sidebar.markdown("---")

# Installation
with st.sidebar.expander("📦 Installation Instructions for PyFock"):
    st.code("""
# Install LibXC — required by PyFock
# For Python < 3.10:
# Install system LibXC and then install pylibxc2 via pip
sudo apt-get install libxc-dev     # Ubuntu/Debian
pip install pylibxc2

# For Python >= 3.10:
# pip wheels for pylibxc2 may be unavailable
# Use conda-forge instead (recommended)
conda install -c conda-forge pylibxc -y

# Install PyFock
pip install pyfock

# Optional: GPU support
pip install cupy-cuda12x   # choose version appropriate for your CUDA setup

""", language="bash")

st.sidebar.markdown("---")

# Performance highlights
with st.sidebar.expander("⚡ Performance Highlights"):
    st.markdown("""
**CPU Performance:**
- Upto 2x faster than PySCF
- Strong scaling up to 32 cores
- ~O(N²·⁰⁵) scaling with basis functions
- Suitable for large systems (upto ~10,000 basis functions)

**GPU Acceleration:**
- Up to **14× speedup** on A100 GPU vs 4-core CPU
- Single A100 GPU handles 4000+ basis functions
- Consumer GPUs (RTX series) supported
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Made with PyFock by PhysWhiz*")
st.sidebar.markdown("*Pure Python • Numba JIT • GPU Ready*")

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

}

# XC functional mapping to libxc codes
XC_FUNCTIONALS = {
    "LDA (SVWN5)": (1, 7),  # Slater exchange + VWN5 correlation
    "PBE": (101, 130),       # PBE exchange + PBE correlation
    "BLYP": (106, 131),      # Becke88 exchange + LYP correlation
    "BP86": (106, 132),      # Becke88 exchange + P86 correlation
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
            "Water", # correct
            "Acetone", # correct
            "Tetrahydrofuran", # correct
            "Pyrrole", # correct
            "Dimethyl ether", # correct
            # "Acetonitrile", 
            # "Methane",
            "Benzene", # correct
            # "Ammonia",
            "Carbon Dioxide", # correct
            "Hydrogen Peroxide",
            # "Formaldehyde",
            # "Hydrogen Cyanide",
            # "Acetylene",
            # "Ethylene",
            # "Ethane",
            "Formic Acid",
            "Hydrogen Sulfide", # correct
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
    
    max_iterations = st.number_input("Max Iterations:", min_value=1, max_value=16, value=14)
    conv_crit = st.number_input("Convergence Criterion:", min_value=1e-7, max_value=1e-3, value=1e-6, format="%.1e")
    ncores = 1#st.number_input("Number of Cores:", min_value=1, max_value=8, value=4)
    use_pyscf_grids = st.checkbox("Use PySCF Grids for XC Term", value=True, help="Use either PySCF grids or PyFock grids for DFT calculaiton. Using PySCF grids is recommended as those are relatively smaller. NOTE: This does not perform a PySCF DFT calculation, only grid generation.") 
    compare_pyscf = st.checkbox("Compare energy with PySCF (will take longer)", value=False, help="Runs a KS-DFT calculation using same settings in PySCF for energy comparison.")

st.markdown("---")

# Visualization settings
st.header("2. Cube Generation and Visualization Settings")
col3, col4, col5 = st.columns(3)

with col3:
    cube_resolution = st.slider("Cube File Resolution (nx=ny=nz):", 30, 50, 40)
with col4:
    isovalue = st.number_input("Isovalue:", 0.0, 1.0, value=0.05, step=0.001, format="%.6f")
with col5:
    opacity = st.slider("Opacity:", 0.0, 1.0, value=0.90, step=0.01)



st.markdown("---")

# Run calculation button
if st.button("🚀 Run DFT Calculation", type="primary"):
    
    # Validate XYZ input
    if not xyz_content.strip():
        st.error("Please provide XYZ coordinates!")
        st.stop()
    
    
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Capture stdout/stderr into buffer and show later in the app
        log_buffer = _io.StringIO()
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            # Setup environment variables
            status_text.text("Setting up environment...")
            progress_bar.progress(5)
            
            os.environ['OMP_NUM_THREADS'] = str(ncores)
            os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
            os.environ["MKL_NUM_THREADS"] = str(ncores)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
            os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
            
            # Import PyFock modules
            status_text.text("Importing PyFock modules...")
            progress_bar.progress(10)
            
            from pyfock import Basis, Mol, DFT, Utils, Grids
            
            # Create temporary XYZ file
            status_text.text("Creating molecule object...")
            progress_bar.progress(15)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
                f.write(xyz_content)
                xyz_file = f.name
            
            # Initialize molecule
            mol = Mol(coordfile=xyz_file)
            
            # Initialize basis sets
            status_text.text(f"Loading basis set: {basis_set}...")
            progress_bar.progress(20)
            
            basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set)})
            auxbasis_obj = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasis)})
            
            # Check actual basis function count
            n_basis = basis.bfs_nao
            if n_basis > 120:
                st.error(f"❌ This system has {n_basis} basis functions, exceeding the limit of 120. Please use a smaller basis set or fewer atoms.")
                os.unlink(xyz_file)
                st.stop()
            
            st.info(f"✓ System has {n_basis} basis functions (within limit)")
            
            # Get XC functional codes
            funcx, funcc = XC_FUNCTIONALS[xc_functional]
            funcidcrysx = [funcx, funcc]
            funcidpyscf = f"{funcx},{funcc}"
            
            # Generating grids
            status_text.text("Generating numerical grids...")
            progress_bar.progress(22)
            if use_pyscf_grids:
                # PySCF grids
                molPySCF = gto.Mole()
                molPySCF.atom = xyz_file
                molPySCF.basis = basis_set
                molPySCF.cart = True
                molPySCF.verbose = 0
                molPySCF.build()
                grids = gen_grid.Grids(molPySCF)
                grids.level = 3        # optional: quality of grid (0–9 approx)
                grids.prune = None     # disable pruning if you want the full Lebedev mesh
                grids.build(with_non0tab=True)
            else:
                # PyFock grids
                grids = Grids(mol, basis=basis, level = 3, radial_precision=1.0e-13, ncores=ncores)

            

            # Initialize DFT object
            status_text.text("Initializing DFT calculation...")
            progress_bar.progress(25)
            
            dftObj = DFT(mol, basis, auxbasis_obj, xc=funcidcrysx, grids=grids, gridsLevel=3)
            dftObj.conv_crit = conv_crit
            dftObj.max_itr = max_iterations
            dftObj.ncores = ncores
            dftObj.save_ao_values = True
            
            # Run SCF calculation
            status_text.text("Running SCF calculation... This may take a few moments.")
            progress_bar.progress(30)
            
            start_time = time.time()
            energyPyFock, dmat = dftObj.scf()
            pyfock_time = time.time() - start_time
            
            progress_bar.progress(50)
            
            # Display results
            if dftObj.converged:
                st.success(f"✅ PyFock KS-DFT calculation converged in {pyfock_time:.2f} seconds and {dftObj.niter} iterations!")
            else:
                st.warning(f"⚠️ PyFock KS-DFT calculation did not converge in {dftObj.niter} iterations and {pyfock_time:.2f} seconds!")
            
            st.header("3. Results")
            
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
                            "Nuclear Repulsion"
                        ],
                        "Energy (Ha)": [
                            f"{dftObj.Kinetic_energy:.8f}",
                            f"{dftObj.Nuc_energy:.8f}",
                            f"{dftObj.J_energy:.8f}",
                            f"{dftObj.XC_energy:.8f}",
                            f"{dftObj.Nuclear_repulsion_energy:.8f}"
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

            # Generate cube files
            status_text.text("Generating cube files for visualization...")
            progress_bar.progress(60)
            
            cube_files = {}
            
            if homo_idx is not None:
                # HOMO cube
                status_text.text("Generating HOMO cube file...")
                with tempfile.NamedTemporaryFile(mode='w', suffix='_HOMO.cube', delete=False) as f:
                    homo_cube_file = f.name
                
                Utils.write_orbital_cube(
                    mol, basis, dftObj.mo_coefficients[:, homo_idx],
                    homo_cube_file, nx=cube_resolution, ny=cube_resolution, nz=cube_resolution,
                    ncores=ncores
                )
                
                with open(homo_cube_file, 'r') as f:
                    cube_files['HOMO'] = f.read()
                
                progress_bar.progress(70)
            
            if lumo_idx is not None:
                # LUMO cube
                status_text.text("Generating LUMO cube file...")
                with tempfile.NamedTemporaryFile(mode='w', suffix='_LUMO.cube', delete=False) as f:
                    lumo_cube_file = f.name
                
                Utils.write_orbital_cube(
                    mol, basis, dftObj.mo_coefficients[:, lumo_idx],
                    lumo_cube_file, nx=cube_resolution, ny=cube_resolution, nz=cube_resolution,
                    ncores=ncores
                )
                
                with open(lumo_cube_file, 'r') as f:
                    cube_files['LUMO'] = f.read()
                
                progress_bar.progress(80)
            
            # Density cube
            status_text.text("Generating electron density cube file...")
            with tempfile.NamedTemporaryFile(mode='w', suffix='_density.cube', delete=False) as f:
                density_cube_file = f.name
            
            Utils.write_density_cube(
                mol, basis, dftObj.dmat,
                density_cube_file, nx=cube_resolution, ny=cube_resolution, nz=cube_resolution,
                ncores=ncores
            )
            
            with open(density_cube_file, 'r') as f:
                cube_files['Density'] = f.read()
            
            progress_bar.progress(85)
            
            
            
            # Display visualizations side-by-side
            st.subheader("4. Visualizations")
            # create columns for HOMO, LUMO and Density (as available)
            vis_cols = []
            num_vis = len([k for k in cube_files.keys() if cube_files.get(k)])
            if num_vis == 0:
                st.info("No cube visualizations available.")
            else:
                # Arrange into up to three columns side-by-side
                if 'HOMO' in cube_files and 'LUMO' in cube_files and 'Density' in cube_files:
                    c1, c2, c3 = st.columns(3)
                    vis_cols = [c1, c2, c3]
                    mapping = [('HOMO', c1), ('LUMO', c2), ('Density', c3)]
                else:
                    # pack present visualizations into equal columns
                    keys_present = list(cube_files.keys())
                    cols = st.columns(len(keys_present))
                    vis_cols = cols
                    mapping = list(zip(keys_present, cols))
                
                for title, col in mapping:
                    if title in cube_files:
                        with col:
                            st.markdown(f"#### {title}")
                            html_blob = visualize_cube_in_component(cube_files[title], title, isovalue, opacity)
                            components.html(html_blob, height=380, width=420)

            col14, col15, col16 = st.columns(3)
            
            # Helper to create base64 download links (avoids widget-triggered reruns)
            def make_download_link(content, filename, mimetype="text/plain"):
                if isinstance(content, str):
                    b = content.encode()
                else:
                    b = content
                b64 = base64.b64encode(b).decode()
                return f'<a href="data:{mimetype};base64,{b64}" download="{filename}">📥 Download {filename}</a>'
            
            with col14:
                if 'HOMO' in cube_files:
                    st.markdown(make_download_link(cube_files['HOMO'], "homo.cube"), unsafe_allow_html=True)
            
            with col15:
                if 'LUMO' in cube_files:
                    st.markdown(make_download_link(cube_files['LUMO'], "lumo.cube"), unsafe_allow_html=True)
            
            with col16:
                if 'Density' in cube_files:
                    st.markdown(make_download_link(cube_files['Density'], "density.cube"), unsafe_allow_html=True)
            
            progress_bar.progress(90)
            
            # === Allow visualizing any orbital ===
            @st.fragment
            def func_viz_any_mo():
                st.subheader("Visualize any MO")
                mo_idx_choice = None
                if hasattr(dftObj, 'mo_energies'):
                    max_orb = len(dftObj.mo_energies) - 1
                    # Show a slider/selectbox to pick orbital
                    mo_idx_choice = st.number_input("Select orbital index to visualize:", min_value=0, max_value=max_orb, value=homo_idx if homo_idx is not None else 0, step=1)
                    # if st.button("Generate and Show Selected MO", key="gen_orb_btn"):
                    status_text.text(f"Generating cube for MO index {mo_idx_choice} ...")
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'_MO{mo_idx_choice}.cube', delete=False) as f:
                        mo_cube_file = f.name
                    Utils.write_orbital_cube(
                        mol, basis, dftObj.mo_coefficients[:, int(mo_idx_choice)],
                        mo_cube_file, nx=cube_resolution, ny=cube_resolution, nz=cube_resolution,
                        ncores=ncores
                    )
                    with open(mo_cube_file, 'r') as f:
                        cube_files[f"MO_{mo_idx_choice}"] = f.read()
                    # show it in a small area
                    html_blob = visualize_cube_in_component(cube_files[f"MO_{mo_idx_choice}"], f"MO {mo_idx_choice}", isovalue, opacity)
                    st.markdown(f"#### MO {mo_idx_choice}")
                    components.html(html_blob, height=380, width=420)
            
            func_viz_any_mo()
                
            
            # PySCF comparison
            if compare_pyscf:
                status_text.text("Running PySCF calculation for comparison...")
                
                try:
                    if not use_pyscf_grids:
                        molPySCF = gto.Mole()
                        molPySCF.atom = xyz_file
                        molPySCF.basis = basis_set
                        molPySCF.cart = True
                        molPySCF.verbose = 5
                        molPySCF.build()
                    
                    mf = dft.rks.RKS(molPySCF).density_fit(auxbasis=auxbasis)
                    mf.xc = funcidpyscf
                    # mf.direct_scf = False
                    mf.conv_tol = conv_crit
                    mf.max_cycle = max_iterations
                    # mf.grids.level = 5
                    mf.grids.coords = grids.coords
                    mf.grids.weights = grids.weights

                    # Disable PySCF's automatic grid generation
                    mf.grids.build = lambda *args, **kwargs: None
                    
                    start_pyscf = time.time()
                    energyPySCF = mf.kernel(dm0 = mf.init_guess_by_1e(molPySCF))
                    pyscf_time = time.time() - start_pyscf
                    
                    st.subheader("5. Comparison with PySCF")
                    
                    col11, col12, col13 = st.columns(3)
                    
                    with col11:
                        st.metric("PySCF Energy", f"{energyPySCF:.8f} Ha")
                    
                    with col12:
                        energy_diff = abs(energyPyFock - energyPySCF) * 1000  # in mHa
                        st.metric("Energy Difference", f"{energy_diff:.6f} mHa")
                    
                    with col13:
                        speedup = pyscf_time / pyfock_time
                        # st.metric("PyFock Speedup", f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower")
                
                        # st.write(f"**PyFock time:** {pyfock_time:.2f} s | **PySCF time:** {pyscf_time:.2f} s")
                    
                except Exception as e:
                    st.warning(f"PySCF comparison failed: {str(e)}")
            
            progress_bar.progress(95)
            
            # Downloads section
            st.subheader("6. INPUT Script Generation")
            
            
            
            
            
            
            
            # Generate Python script
            status_text.text("Generating Python script...")
            
            python_script = """# PyFock DFT Calculation Script
# Generated by PyFock GUI

import os
ncores = {ncores}
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, Mol, DFT, Utils
import numpy as np

# XYZ coordinates
xyz_content = \"\"\"
{xyz_content}
\"\"\"

# Save XYZ to file
with open('molecule.xyz', 'w') as f:
    f.write(xyz_content)

# Calculation parameters
basis_set_name = '{basis_set}'
auxbasis_name = '{auxbasis}'
funcx = {funcx}
funcc = {funcc}
funcidcrysx = [funcx, funcc]

# Initialize molecule and basis
mol = Mol(coordfile='molecule.xyz')
basis = Basis(mol, {{'all': Basis.load(mol=mol, basis_name=basis_set_name)}})
auxbasis = Basis(mol, {{'all': Basis.load(mol=mol, basis_name=auxbasis_name)}})

# Setup DFT calculation
dftObj = DFT(mol, basis, auxbasis, xc=funcidcrysx)
dftObj.conv_crit = {conv_crit}
dftObj.max_itr = {max_iterations}
dftObj.ncores = ncores
dftObj.save_ao_values = True

# Run SCF
energy, dmat = dftObj.scf()

print(f"Total Energy: {{energy:.8f}} Ha")

# Find HOMO and LUMO
occupied = np.where(dftObj.mo_occupations > 1e-8)[0]
if len(occupied) > 0 and len(occupied) < len(dftObj.mo_energies):
    homo_idx = occupied[-1]
    lumo_idx = homo_idx + 1
    
    # Generate cube files
    Utils.write_orbital_cube(mol, basis, dftObj.mo_coefficients[:, homo_idx], 
                            'HOMO.cube', nx={cube_resolution}, ny={cube_resolution}, 
                            nz={cube_resolution}, ncores=ncores)
    
    Utils.write_orbital_cube(mol, basis, dftObj.mo_coefficients[:, lumo_idx], 
                            'LUMO.cube', nx={cube_resolution}, ny={cube_resolution}, 
                            nz={cube_resolution}, ncores=ncores)

# Generate density cube
Utils.write_density_cube(mol, basis, dftObj.dmat, 'density.cube', 
                        nx={cube_resolution}, ny={cube_resolution}, 
                        nz={cube_resolution}, ncores=ncores)

print("Cube files generated successfully!")
"""
            parameters = {
                'ncores': ncores,
                'xyz_content': xyz_content,
                'basis_set': basis_set,
                'auxbasis': auxbasis, # Often a different basis set for auxiliary functions
                'funcx': funcx,        # Example ID for exchange functional (like LDA_X)
                'funcc': funcc,        # Example ID for correlation functional (like LDA_C_PZ)
                'conv_crit': conv_crit,
                'max_iterations': max_iterations,
                'cube_resolution': cube_resolution
            }
            python_script = python_script.format(**parameters)
            with st.expander("Input Script to Run the Above PyFock Calculation"):
                st.code(python_script, language='python')
            # Provide script as base64 link (no rerun on click)
            st.markdown(make_download_link(python_script, "pyfock_calculation.py", mimetype="text/x-python"), unsafe_allow_html=True)
            
            progress_bar.progress(100)
            status_text.text("✅ All tasks completed!")
            
            # === Show captured stdout/stderr logs ===
            st.subheader("7. Calculation Log Output")
            log_buffer.seek(0)
            log_text = log_buffer.read()
            if log_text.strip():
                st.code(strip_ansi(log_text))
            else:
                st.write("No log output was captured.")
            
            
            # Cleanup
            os.unlink(xyz_file)
            if 'homo_cube_file' in locals():
                os.unlink(homo_cube_file)
            if 'lumo_cube_file' in locals():
                os.unlink(lumo_cube_file)
            if 'density_cube_file' in locals():
                os.unlink(density_cube_file)
            if 'mo_cube_file' in locals():
                try:
                    os.unlink(mo_cube_file)
                except Exception:
                    pass

    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.info("Make sure PyFock is installed: `pip install pyfock`")
        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        st.error(f"❌ Calculation failed: {str(e)}")
        st.info("Please check your input parameters and try again.")
        progress_bar.empty()
        status_text.empty()
        
        # Cleanup on error
        if 'xyz_file' in locals():
            try:
                os.unlink(xyz_file)
            except:
                pass

else:
    # Initial instructions
    st.info("👆 Configure your calculation parameters above and click 'Run DFT Calculation' to start!")
    
    st.markdown("""
    ### Getting Started
    
    1. **Choose a molecule**: Select from examples or paste your own XYZ coordinates
    2. **Select calculation parameters**: Basis set, functional, convergence criteria
    3. **Adjust visualization settings**: Cube resolution, isovalue, opacity
    4. **Run calculation**: Click the button above
    5. **Explore results**: View energies, MO properties, and 3D visualizations
    6. **Download**: Get cube files and a Python script to reproduce the calculation
    
    ### Notes
    
    - Calculations are limited to ~120 basis functions when running on cloud. Download the repo and run the streamlit app locally for larger systems.
    - Smaller molecules and basis sets will run faster.
    - PySCF comparison adds computation time but validates energy.
    - All calculations use density fitting (resolution of identity) for efficiency.
    
    ### Example Systems
    
    - **Water**: Quick test system (7 basis functions with sto-3g)
    - **Benzene**: Aromatic system (42 basis functions with sto-3g)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>PyFock GUI - Pure Python DFT with Numba JIT acceleration</p>
    <p>⚡ Fast • 🎯 Accurate • 🐍 Pure Python</p>
</div>
""", unsafe_allow_html=True)
