import streamlit as st


PYFOCK_PAPER_TITLE = "PyFock: A Just-In-Time Compiled Gaussian Basis DFT Python Code for CPU and GPU Architectures"
PYFOCK_PAPER_AUTHORS = "Manas Sharma and Marek Sierka"
PYFOCK_PAPER_DATE = "26 May 2026"
CHEMRXIV_DOI = "10.26434/chemrxiv.15003943/v1"
CHEMRXIV_URL = f"https://doi.org/{CHEMRXIV_DOI}"


def set_background_video_css():
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )


def embed_background_video():
    video_link = "https://raw.githubusercontent.com/manassharma07/Website_Files_for_PyFock/main/background_video_pyfock.mp4"
    st.sidebar.markdown(
        f"""
        <video autoplay muted loop id="myVideo">
            <source src="{video_link}">
            Your browser does not support HTML5 video.
        </video>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    set_background_video_css()
    embed_background_video()

    st.logo(
        "logo_crysx_pyfock_dark_mode_landscape_transparent.png",
        size="large",
        link="https://github.com/manassharma07/pyfock",
    )

    st.sidebar.markdown("### About PyFock")
    st.sidebar.markdown(
        """
**Pure Python DFT** with performance matching C++ codes!

**Key Advantages:**
- 100% Pure Python (including molecular integrals and XC evaluation)
- Numba JIT acceleration
- GPU support (CUDA via CuPy)
- Near-quadratic scaling (~O(N²))
- Accuracy matching PySCF (<10⁻⁷ Ha)
- Windows/Linux/MacOS compatible
- Easy pip installation
"""
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### GUI Features")
    st.sidebar.markdown(
        """
* Run DFT in your browser
* Visualize HOMO, LUMO, density
* Compare with PySCF
* Download cube files & scripts
* Interactive 3D visualization
* Calculate molecular integrals
* No installation required!
"""
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🔗 Links & Resources")
    st.sidebar.markdown(
        f"""
[![GitHub (PyFock)](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/manassharma07/PyFock)
[![GitHub (PyFock GUI)](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/manassharma07/PyFock-GUI)
[![PyPI](https://img.shields.io/badge/PyPI-Package-orange?logo=pypi)](https://pypi.org/project/pyfock/)
[![Docs](https://img.shields.io/badge/Documentation-Read-green?logo=readthedocs)](https://pyfock-docs.bragitoff.com)
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-Preprint-red)]({CHEMRXIV_URL})

📄 **Preprint:**
**{PYFOCK_PAPER_TITLE}**
{PYFOCK_PAPER_AUTHORS}, *ChemRxiv*, {PYFOCK_PAPER_DATE}.

🔗 **DOI:** [{CHEMRXIV_DOI}]({CHEMRXIV_URL})

👨‍💻 **Developer:** [Manas Sharma](https://www.linkedin.com/in/manassharma07)

⭐ **Star the repo** if you find it useful!
"""
    )

    st.sidebar.markdown("---")

    with st.sidebar.expander("📦 Installation Instructions for PyFock"):
        st.code(
            """
# Install PyFock
pip install pyfock

# Optional: GPU support
pip install cupy-cuda12x   # choose version appropriate for your CUDA setup

# Optional: LibXC — required for more DFT functionals
# For Python < 3.10:
# Install system LibXC and then install pylibxc2 via pip
sudo apt-get install libxc-dev     # Ubuntu/Debian
pip install pylibxc2

# For Python >= 3.10:
# pip wheels for pylibxc2 may be unavailable
# Use conda-forge instead (recommended)
conda install -c conda-forge pylibxc -y
""",
            language="bash",
        )

    st.sidebar.markdown("---")

    with st.sidebar.expander("⚡ Performance Highlights"):
        st.markdown(
            """
**CPU Performance:**
- Upto 2x faster than PySCF
- Good scaling up to 32 cores
- ~O(N²) scaling with basis functions
- Suitable for large systems (upto ~20,000 basis functions)

**GPU Acceleration:**
- Up to **24× speedup** on A100 GPU vs 4-core CPU
- Single A100 GPU handles ~4000 basis functions
- Consumer GPUs (RTX series) supported
"""
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Pure Python • Numba JIT • GPU Ready*")
