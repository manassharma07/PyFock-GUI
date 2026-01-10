import streamlit as st
import os
import tempfile
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
from io import StringIO
import time
from ase import Atoms
from ase.io import read as ase_read
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title='PyFock GUI - Molecular Integrals',
    layout='wide',
    page_icon="⚛️",
    menu_items={
        'About': "PyFock GUI - A web interface for PyFock, a pure Python DFT code with Numba JIT acceleration"
    }
)

# === Background video styling ===
def set_css():
    st.markdown("""
        <style>
            #myVideo {
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%; 
                min-height: 100%;
                opacity: 0.12;
                pointer-events: none;
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

# Sidebar with enhanced styling (same as home page)
st.sidebar.image("https://raw.githubusercontent.com/manassharma07/PyFock/main/logo_crysx_pyfock.png", use_container_width=True)
st.sidebar.markdown("---")

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

with st.sidebar.expander("📦 Installation Instructions for PyFock"):
    st.code("""
# Install LibXC — required by PyFock
# For Python < 3.10:
sudo apt-get install libxc-dev     # Ubuntu/Debian
pip install pylibxc2

# For Python >= 3.10:
conda install -c conda-forge pylibxc -y

# Install PyFock
pip install pyfock

# Optional: GPU support
pip install cupy-cuda12x
""", language="bash")

st.sidebar.markdown("---")

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

# Helper functions
def _parse_xyz_to_atoms(xyz_text):
    return ase_read(StringIO(xyz_text), format='xyz')

def get_structure_viz2(atoms_obj, style='stick', width=400, height=400):
    xyz_str = ""
    xyz_str += f"{len(atoms_obj)}\n"
    xyz_str += "Structure\n"
    for atom in atoms_obj:
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
    view.zoomTo()
    view.setBackgroundColor('white')
    return view

# Example molecules (same as home page)
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

    "Hydrogen Sulfide": """3
Hydrogen sulfide molecule
S     0.000000    0.000000    0.000000
H     0.960000    0.000000    0.000000
H    -0.480000    0.830000    0.000000""",
}

BASIS_SETS = ["sto-3g", "sto-6g", "3-21G", "4-31G", "6-31G", "6-31+G", "6-31++G", "cc-pvDZ", "def2-SVP", "def2-TZVP"]

# Main title
st.title("🧮 PyFock - Molecular Integrals Calculator")
st.markdown("""
Explore the fundamental quantum mechanical integrals that form the basis of electronic structure calculations.
Learn how overlap, kinetic, nuclear attraction, and electron repulsion integrals are computed!
""")
st.markdown("---")

# Input section
st.header("1. System Setup")

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("Molecule Input")
    
    molecule_choice = st.selectbox(
        "Select example molecule or paste custom XYZ:",
        [
            "Water",
            "Acetone",
            "Methane",
            "Benzene",
            "Ammonia",
            "Carbon Dioxide",
            "Hydrogen Peroxide",
            "Formaldehyde",
            "Hydrogen Sulfide",
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

with col2:
    if xyz_content and xyz_content.strip():
        st.markdown("### Molecule Visualization")
        viz_style = st.selectbox("Select Visualization Style:", ["ball-stick", "stick", "ball"], key="viz_style_select")
        atoms_obj = _parse_xyz_to_atoms(xyz_content)
        
        view_3d = get_structure_viz2(atoms_obj, style=viz_style, width=400, height=400)
        try:
            st.components.v1.html(view_3d._make_html(), width=420, height=420)
        except Exception:
            t = view_3d.js()
            html_content = f"{t.startjs}{t.endjs}"
            components.html(html_content, height=420, width=420)

        st.markdown("### Structure Information")
        atoms_info = {
            "Number of Atoms": len(atoms_obj),
            "Chemical Formula": atoms_obj.get_chemical_formula() if hasattr(atoms_obj, 'get_chemical_formula') else "".join(atoms_obj.get_chemical_symbols()),
            "Atom Types": ", ".join(sorted(list(set(atoms_obj.get_chemical_symbols()))))
        }
        
        for key, value in atoms_info.items():
            st.write(f"**{key}:** {value}")

with col1:
    st.subheader("Basis Set Configuration")
    basis_set = st.selectbox("Basis Set:", BASIS_SETS, index=0)
    use_spherical = st.checkbox("Convert to Spherical AOs (SAO)", value=False, 
                                help="Convert integrals from Cartesian AOs (CAO) to Spherical AOs (SAO)")

st.markdown("---")

# Integral selection
st.header("2. Select Integrals to Calculate")

col3, col4 = st.columns(2)

with col3:
    st.subheader("One-Electron Integrals")
    calc_overlap = st.checkbox("Overlap Integrals (S)", value=True,
                               help="⟨φᵢ|φⱼ⟩ - Measures orbital overlap")
    calc_kinetic = st.checkbox("Kinetic Energy Integrals (T)", value=True,
                               help="⟨φᵢ|-½∇²|φⱼ⟩ - Kinetic energy operator")
    calc_nuclear = st.checkbox("Nuclear Attraction Integrals (V)", value=True,
                               help="⟨φᵢ|-Σ Zₐ/rₐ|φⱼ⟩ - Electron-nuclear attraction")

with col4:
    st.subheader("Two-Electron Integrals")
    calc_eri_4c2e = st.checkbox("4-Center 2-Electron (ERI)", value=False,
                                help="⟨φᵢφⱼ|1/r₁₂|φₖφₗ⟩ - Electron-electron repulsion")
    
    if calc_eri_4c2e:
        eri_algorithm = st.radio("Algorithm:", ["Rys Quadrature (Fast)", "Conventional (Slow)"],
                                help="Rys quadrature is significantly faster for most systems")
    
    calc_eri_3c2e = st.checkbox("3-Center 2-Electron (3c2e)", value=False,
                                help="Used in density fitting / RI approximations")
    calc_eri_2c2e = st.checkbox("2-Center 2-Electron (2c2e)", value=False,
                                help="Auxiliary basis integrals for density fitting")

# Subset selection
st.subheader("Optional: Calculate Subset of Matrix")
use_subset = st.checkbox("Calculate only a subset of the integral matrix", value=False)

if use_subset:
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        row_start = st.number_input("Row Start:", min_value=0, value=0, step=1)
    with col6:
        row_end = st.number_input("Row End:", min_value=1, value=5, step=1)
    with col7:
        col_start = st.number_input("Col Start:", min_value=0, value=0, step=1)
    with col8:
        col_end = st.number_input("Col End:", min_value=1, value=5, step=1)

st.markdown("---")

# Calculate button
if st.button("🧮 Calculate Integrals", type="primary"):
    
    if not xyz_content.strip():
        st.error("Please provide XYZ coordinates!")
        st.stop()
    
    # Check if at least one integral type is selected
    if not any([calc_overlap, calc_kinetic, calc_nuclear, calc_eri_4c2e, calc_eri_3c2e, calc_eri_2c2e]):
        st.error("Please select at least one integral type to calculate!")
        st.stop()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Importing PyFock modules...")
        progress_bar.progress(10)
        
        from pyfock import Basis, Mol, Integrals
        
        status_text.text("Creating molecule object...")
        progress_bar.progress(20)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write(xyz_content)
            xyz_file = f.name
        
        mol = Mol(coordfile=xyz_file)
        
        status_text.text(f"Loading basis set: {basis_set}...")
        progress_bar.progress(30)
        
        basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set)})
        
        n_basis = basis.bfs_nao
        st.info(f"✓ System has {n_basis} basis functions")
        if n_basis > 50:
            st.error(f"❌ This system has {n_basis} basis functions, exceeding the limit of 50. Please use a smaller basis set or fewer atoms.")
            os.unlink(xyz_file)
            st.stop()
        
        # Prepare results storage
        results = {}
        timings = {}
        
        # Setup subset slice if needed
        if use_subset:
            subset_slice = [row_start, row_end, col_start, col_end]
        else:
            subset_slice = None
        
        progress_step = 40
        progress_increment = 50 / sum([calc_overlap, calc_kinetic, calc_nuclear, 
                                       calc_eri_4c2e, calc_eri_3c2e, calc_eri_2c2e])
        
        # Calculate one-electron integrals
        if calc_overlap:
            status_text.text("Calculating overlap integrals...")
            start = time.time()
            if subset_slice:
                S = Integrals.overlap_mat_symm(basis, slice=subset_slice)
            else:
                S = Integrals.overlap_mat_symm(basis)
            
            if use_spherical:
                c2sph_mat = basis.cart2sph_basis()
                S = np.dot(c2sph_mat, np.dot(S, c2sph_mat.T))
            
            results['Overlap'] = S
            timings['Overlap'] = time.time() - start
            progress_step += progress_increment
            progress_bar.progress(int(progress_step))
        
        if calc_kinetic:
            status_text.text("Calculating kinetic energy integrals...")
            start = time.time()
            if subset_slice:
                T = Integrals.kin_mat_symm(basis, slice=subset_slice)
            else:
                T = Integrals.kin_mat_symm(basis)
            
            if use_spherical:
                c2sph_mat = basis.cart2sph_basis()
                T = np.dot(c2sph_mat, np.dot(T, c2sph_mat.T))
            
            results['Kinetic'] = T
            timings['Kinetic'] = time.time() - start
            progress_step += progress_increment
            progress_bar.progress(int(progress_step))
        
        if calc_nuclear:
            status_text.text("Calculating nuclear attraction integrals...")
            start = time.time()
            if subset_slice:
                V = Integrals.nuc_mat_symm(basis, mol, slice=subset_slice)
            else:
                V = Integrals.nuc_mat_symm(basis, mol)
            
            if use_spherical:
                c2sph_mat = basis.cart2sph_basis()
                V = np.dot(c2sph_mat, np.dot(V, c2sph_mat.T))
            
            results['Nuclear'] = V
            timings['Nuclear'] = time.time() - start
            progress_step += progress_increment
            progress_bar.progress(int(progress_step))
        
        # Calculate two-electron integrals
        if calc_eri_4c2e:
            if eri_algorithm == "Rys Quadrature (Fast)":
                status_text.text("Calculating 4c2e integrals (Rys quadrature)...")
                start = time.time()
                ERI = Integrals.rys_4c2e_symm(basis)
                timings['4c2e (Rys)'] = time.time() - start
                results['4c2e'] = ERI
            else:
                status_text.text("Calculating 4c2e integrals (Conventional - this may take a while)...")
                start = time.time()
                ERI = Integrals.conv_4c2e_symm(basis)
                timings['4c2e (Conv)'] = time.time() - start
                results['4c2e'] = ERI
            
            progress_step += progress_increment
            progress_bar.progress(int(progress_step))
        
        if calc_eri_3c2e:
            status_text.text("Calculating 3c2e integrals...")
            start = time.time()
            ERI_3c = Integrals.rys_3c2e_symm(basis)
            results['3c2e'] = ERI_3c
            timings['3c2e'] = time.time() - start
            progress_step += progress_increment
            progress_bar.progress(int(progress_step))
        
        if calc_eri_2c2e:
            status_text.text("Calculating 2c2e integrals...")
            start = time.time()
            ERI_2c = Integrals.rys_2c2e_symm(basis)
            results['2c2e'] = ERI_2c
            timings['2c2e'] = time.time() - start
            progress_step += progress_increment
            progress_bar.progress(int(progress_step))
        
        progress_bar.progress(100)
        status_text.text("✅ All calculations completed!")
        
        st.success("✅ Integral calculations completed successfully!")
        
        # Display results
        st.header("3. Results")
        
        # Timing summary
        st.subheader("Computation Times")
        timing_cols = st.columns(len(timings))
        for idx, (name, timing) in enumerate(timings.items()):
            with timing_cols[idx]:
                st.metric(name, f"{timing:.4f} s")
        
        st.markdown("---")
        
        # Display each integral type
        for integral_name, integral_matrix in results.items():
            st.subheader(f"{integral_name} Integrals")
            
            # Educational information
            with st.expander(f"ℹ️ About {integral_name} Integrals"):
                if integral_name == "Overlap":
                    st.markdown("""
                    **Overlap Integrals (S)**
                    
                    The overlap integral measures how much two basis functions overlap in space:
                    
                    $$S_{ij} = \\langle \\phi_i | \\phi_j \\rangle = \\int \\phi_i(\\mathbf{r}) \\phi_j(\\mathbf{r}) d\\mathbf{r}$$
                    
                    - Diagonal elements (Sᵢᵢ) equal 1 for normalized basis functions
                    - Off-diagonal elements indicate orbital overlap
                    - Essential for orthogonalization and transformation to orthonormal basis
                    - Used in Löwdin or canonical orthogonalization schemes
                    """)
                elif integral_name == "Kinetic":
                    st.markdown("""
                    **Kinetic Energy Integrals (T)**
                    
                    Represents the kinetic energy operator in the basis function representation:
                    
                    $$T_{ij} = \\langle \\phi_i | -\\frac{1}{2}\\nabla^2 | \\phi_j \\rangle$$
                    
                    - Contains the electronic kinetic energy contribution
                    - Always positive (kinetic energy is positive definite)
                    - Diagonal elements are largest (self-kinetic energy)
                    - Critical for total electronic energy calculations
                    """)
                elif integral_name == "Nuclear":
                    st.markdown("""
                    **Nuclear Attraction Integrals (V)**
                    
                    Represents electron-nuclear attraction energy:
                    
                    $$V_{ij} = \\langle \\phi_i | -\\sum_A \\frac{Z_A}{r_A} | \\phi_j \\rangle$$
                    
                    - Sum over all nuclei A with charge Zₐ
                    - Always negative (attractive interaction)
                    - Largest near nuclear positions
                    - Molecular geometry dependent
                    """)
                elif integral_name == "4c2e":
                    st.markdown("""
                    **Four-Center Two-Electron Repulsion Integrals (ERI)**
                    
                    The most computationally expensive integrals in quantum chemistry:
                    
                    $$ERI_{ijkl} = \\langle \\phi_i \\phi_j | \\frac{1}{r_{12}} | \\phi_k \\phi_l \\rangle$$
                    
                    - Four-index tensor: scales as O(N⁴) with basis size
                    - Symmetries reduce unique elements: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)
                    - Used for electron-electron repulsion (Coulomb and exchange)
                    - **Rys quadrature** significantly faster than conventional evaluation
                    """)
                elif integral_name == "3c2e":
                    st.markdown("""
                    **Three-Center Two-Electron Integrals**
                    
                    Used in density fitting (RI) approximations:
                    
                    $$(\\phi_i \\phi_j | P)$$
                    
                    - Three indices instead of four
                    - Auxiliary basis function P
                    - Enables efficient approximation of 4c2e integrals
                    - Critical for linear-scaling DFT methods
                    """)
                elif integral_name == "2c2e":
                    st.markdown("""
                    **Two-Center Two-Electron Integrals**
                    
                    Coulomb metric in auxiliary basis:
                    
                    $$(P | Q)$$
                    
                    - Two-index matrix
                    - Auxiliary basis only
                    - Used in density fitting inverse metric
                    - Much smaller than full ERI tensor
                    """)
            
            # Matrix visualization
            col_a, col_b = st.columns([1.5, 1])
            
            with col_a:
                st.markdown("**Matrix Visualization**")
                
                # Handle different dimensionalities
                if len(integral_matrix.shape) == 2:
                    # 2D matrix (one-electron integrals or 2c2e)
                    fig = px.imshow(integral_matrix, 
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto',
                                   labels={'x': 'Basis Function j', 'y': 'Basis Function i', 'color': 'Value'})
                    fig.update_layout(height=400, title=f"{integral_name} Matrix Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif len(integral_matrix.shape) == 3:
                    # 3D tensor (3c2e)
                    st.info("3D tensor - showing slice along first auxiliary basis index")
                    slice_idx = st.slider(f"Auxiliary basis index:", 0, integral_matrix.shape[0]-1, 0)
                    fig = px.imshow(integral_matrix[slice_idx], 
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto',
                                   labels={'x': 'Basis Function j', 'y': 'Basis Function i', 'color': 'Value'})
                    fig.update_layout(height=400, title=f"{integral_name} Matrix (Slice {slice_idx})")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif len(integral_matrix.shape) == 4:
                    # 4D tensor (4c2e)
                    st.info("4D tensor - showing 2D slice")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        k_idx = st.slider("Index k:", 0, integral_matrix.shape[2]-1, 0, key='k_idx')
                    with col_s2:
                        l_idx = st.slider("Index l:", 0, integral_matrix.shape[3]-1, 0, key='l_idx')
                    
                    fig = px.imshow(integral_matrix[:, :, k_idx, l_idx],
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto',
                                   labels={'x': 'Basis Function j', 'y': 'Basis Function i', 'color': 'Value'})
                    fig.update_layout(height=400, title=f"{integral_name} Matrix (k={k_idx}, l={l_idx})")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                st.markdown("**Matrix Statistics**")
                
                # Calculate statistics based on dimensionality
                if len(integral_matrix.shape) == 2:
                    mat_stats = {
                        "Shape": f"{integral_matrix.shape[0]} × {integral_matrix.shape[1]}",
                        "Max Value": f"{np.max(integral_matrix):.6e}",
                        "Min Value": f"{np.min(integral_matrix):.6e}",
                        "Mean": f"{np.mean(integral_matrix):.6e}",
                        "Std Dev": f"{np.std(integral_matrix):.6e}",
                        "Frobenius Norm": f"{np.linalg.norm(integral_matrix):.6e}"
                    }
                    
                    # Check symmetry for 2D matrices
                    if integral_matrix.shape[0] == integral_matrix.shape[1]:
                        symmetry_error = np.max(np.abs(integral_matrix - integral_matrix.T))
                        mat_stats["Symmetry Error"] = f"{symmetry_error:.6e}"
                        
                        # Check if matrix is positive definite (for overlap)
                        if integral_name == "Overlap":
                            eigenvalues = np.linalg.eigvalsh(integral_matrix)
                            mat_stats["Min Eigenvalue"] = f"{np.min(eigenvalues):.6e}"
                            mat_stats["Condition Number"] = f"{np.max(eigenvalues)/np.min(eigenvalues):.2e}"
                    
                elif len(integral_matrix.shape) == 3:
                    mat_stats = {
                        "Shape": f"{integral_matrix.shape[0]} × {integral_matrix.shape[1]} × {integral_matrix.shape[2]}",
                        "Total Elements": f"{integral_matrix.size:,}",
                        "Max Value": f"{np.max(integral_matrix):.6e}",
                        "Min Value": f"{np.min(integral_matrix):.6e}",
                        "Mean": f"{np.mean(integral_matrix):.6e}",
                        "Memory": f"{integral_matrix.nbytes / 1024:.2f} KB"
                    }
                    
                elif len(integral_matrix.shape) == 4:
                    mat_stats = {
                        "Shape": f"{integral_matrix.shape[0]} × {integral_matrix.shape[1]} × {integral_matrix.shape[2]} × {integral_matrix.shape[3]}",
                        "Total Elements": f"{integral_matrix.size:,}",
                        "Max Value": f"{np.max(integral_matrix):.6e}",
                        "Min Value": f"{np.min(integral_matrix):.6e}",
                        "Mean": f"{np.mean(integral_matrix):.6e}",
                        "Memory": f"{integral_matrix.nbytes / (1024*1024):.2f} MB"
                    }
                    
                    # Note about 8-fold symmetry
                    st.info("**Symmetry**: ERI tensor has 8-fold permutational symmetry")
                
                for key, value in mat_stats.items():
                    st.write(f"**{key}:** {value}")
            
            # Matrix data table (expandable)
            with st.expander(f"📊 View {integral_name} Matrix Data"):
                if len(integral_matrix.shape) == 2:
                    df = pd.DataFrame(integral_matrix)
                    df.columns = [f"j={i}" for i in range(df.shape[1])]
                    df.index = [f"i={i}" for i in range(df.shape[0])]
                    st.dataframe(df, use_container_width=True, height=400)
                else:
                    st.write(f"Shape: {integral_matrix.shape}")
                    st.write("Full tensor too large to display as table. Use visualization above.")
                    
                    # Option to view specific elements
                    st.markdown("**Query Specific Element:**")
                    if len(integral_matrix.shape) == 3:
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            q_i = st.number_input("Index i:", 0, integral_matrix.shape[0]-1, 0, key=f"q_i_{integral_name}")
                        with col_q2:
                            q_j = st.number_input("Index j:", 0, integral_matrix.shape[1]-1, 0, key=f"q_j_{integral_name}")
                        with col_q3:
                            q_k = st.number_input("Index k:", 0, integral_matrix.shape[2]-1, 0, key=f"q_k_{integral_name}")
                        st.write(f"**Value [{q_i},{q_j},{q_k}]:** {integral_matrix[q_i, q_j, q_k]:.8e}")
                    
                    elif len(integral_matrix.shape) == 4:
                        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                        with col_q1:
                            q_i = st.number_input("Index i:", 0, integral_matrix.shape[0]-1, 0, key=f"q_i_{integral_name}")
                        with col_q2:
                            q_j = st.number_input("Index j:", 0, integral_matrix.shape[1]-1, 0, key=f"q_j_{integral_name}")
                        with col_q3:
                            q_k = st.number_input("Index k:", 0, integral_matrix.shape[2]-1, 0, key=f"q_k_{integral_name}")
                        with col_q4:
                            q_l = st.number_input("Index l:", 0, integral_matrix.shape[3]-1, 0, key=f"q_l_{integral_name}")
                        st.write(f"**Value [{q_i},{q_j},{q_k},{q_l}]:** {integral_matrix[q_i, q_j, q_k, q_l]:.8e}")
            
            # Download option
            if len(integral_matrix.shape) == 2:
                csv_data = pd.DataFrame(integral_matrix).to_csv(index=False)
                st.download_button(
                    label=f"💾 Download {integral_name} Matrix (CSV)",
                    data=csv_data,
                    file_name=f"{integral_name.lower()}_integrals.csv",
                    mime="text/csv"
                )
            else:
                # For higher dimensional arrays, save as numpy
                import io
                buf = io.BytesIO()
                np.save(buf, integral_matrix)
                buf.seek(0)
                st.download_button(
                    label=f"💾 Download {integral_name} Tensor (NumPy)",
                    data=buf,
                    file_name=f"{integral_name.lower()}_integrals.npy",
                    mime="application/octet-stream"
                )
            
            st.markdown("---")
        
        # Educational insights section
        if calc_overlap and calc_kinetic and calc_nuclear:
            st.header("4. Educational Insights")
            
            st.subheader("Core Hamiltonian Matrix (H_core)")
            st.markdown("""
            The core Hamiltonian combines kinetic and nuclear attraction integrals:
            
            $H^{\\text{core}}_{ij} = T_{ij} + V_{ij}$
            
            This represents the one-electron part of the Hamiltonian in the Hartree-Fock and DFT methods.
            """)
            
            H_core = results['Kinetic'] + results['Nuclear']
            
            col_h1, col_h2 = st.columns([1.5, 1])
            
            with col_h1:
                fig = px.imshow(H_core,
                               color_continuous_scale='RdBu_r',
                               aspect='auto',
                               labels={'x': 'Basis Function j', 'y': 'Basis Function i', 'color': 'Energy (Ha)'})
                fig.update_layout(height=400, title="Core Hamiltonian Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_h2:
                st.markdown("**Properties:**")
                eigenvalues = np.linalg.eigvalsh(H_core)
                st.write(f"**Lowest eigenvalue:** {np.min(eigenvalues):.6f} Ha")
                st.write(f"**Highest eigenvalue:** {np.max(eigenvalues):.6f} Ha")
                st.write(f"**Energy range:** {np.max(eigenvalues) - np.min(eigenvalues):.6f} Ha")
                
                # Eigenvalue distribution
                fig_eig = go.Figure()
                fig_eig.add_trace(go.Scatter(
                    x=list(range(len(eigenvalues))),
                    y=eigenvalues * 27.2114,  # Convert to eV
                    mode='markers+lines',
                    name='Eigenvalues'
                ))
                fig_eig.update_layout(
                    title="Core Hamiltonian Eigenvalues",
                    xaxis_title="Index",
                    yaxis_title="Energy (eV)",
                    height=300
                )
                st.plotly_chart(fig_eig, use_container_width=True)
        
        # Cleanup
        os.unlink(xyz_file)
        
    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.info("Make sure PyFock is installed: `pip install pyfock`")
        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        st.error(f"❌ Calculation failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()
        
        if 'xyz_file' in locals():
            try:
                os.unlink(xyz_file)
            except:
                pass

else:
    # Initial instructions
    st.info("👆 Configure your molecule and select integrals to calculate above!")
    
    st.markdown("""
    ### About Molecular Integrals
    
    Molecular integrals are the fundamental building blocks of quantum chemistry calculations. This tool allows you to:
    
    1. **Calculate different types of integrals** - from simple overlap to complex four-center electron repulsion integrals
    2. **Visualize integral matrices** - see the structure and patterns in the data
    3. **Learn quantum chemistry** - educational explanations for each integral type
    4. **Compare algorithms** - see the performance difference between Rys quadrature and conventional methods
    5. **Export results** - download matrices for further analysis
    
    ### Types of Integrals
    
    **One-Electron Integrals:**
    - **Overlap (S)**: Measures basis function overlap
    - **Kinetic (T)**: Electronic kinetic energy
    - **Nuclear (V)**: Electron-nuclear attraction
    
    **Two-Electron Integrals:**
    - **4c2e (ERI)**: Four-center electron repulsion - the most computationally intensive
    - **3c2e**: Three-center integrals for density fitting
    - **2c2e**: Two-center auxiliary basis integrals
    
    ### Performance Tips
    
    - Start with small molecules and minimal basis sets (sto-3g)
    - Use **Rys quadrature** for 4c2e integrals (much faster!)
    - 4c2e integrals scale as O(N⁴) - they become very large quickly
    - Consider using subsets for exploration of large systems
    
    ### Educational Use
    
    This tool is perfect for:
    - Learning how basis functions interact
    - Understanding the structure of integral matrices
    - Exploring symmetries in quantum chemistry
    - Teaching computational chemistry concepts
    - Comparing different computational methods
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>PyFock Molecular Integrals Calculator</p>
    <p>⚡ Fast • 🎯 Accurate • 🐍 Pure Python</p>
</div>
""", unsafe_allow_html=True)