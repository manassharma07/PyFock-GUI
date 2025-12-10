# PyFock GUI - Interactive DFT Calculations in Your Browser

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pyfock.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyFock](https://img.shields.io/badge/PyFock-Latest-orange.svg)](https://github.com/manassharma07/PyFock)

A modern, interactive web interface for [PyFock](https://github.com/manassharma07/PyFock) - a pure Python DFT code with Numba JIT acceleration and performance matching C++ implementations.

![PyFock GUI Banner](https://raw.githubusercontent.com/manassharma07/PyFock/main/logo_crysx_pyfock.png)

## Live Demo

Try PyFock GUI instantly without any installation:

- Primary: [https://pyfock.streamlit.app](https://pyfock.streamlit.app)
- Alternative: [https://pyfock-gui.bragitoff.com](https://pyfock-gui.bragitoff.com)
- HuggingFace: [https://manassharma07-pyfock-gui.hf.space/](https://manassharma07-pyfock-gui.hf.space/)

## Features

### Computational Capabilities
Pure Python DFT calculations with performance matching C++ codes. Supports multiple XC functionals (LDA, PBE, BLYP, BP86) and flexible basis sets (STO-3G, 6-31G, cc-pVDZ, def2-SVP, def2-TZVP). Includes density fitting for efficiency and optional PySCF comparison for validation.

### Visualization
Interactive 3D structure viewer with customizable styles. Real-time HOMO/LUMO visualization with adjustable isosurfaces. Electron density maps and exploration of any molecular orbital.

### Input/Output
15+ example molecules included (water, benzene, acetone, pyrrole, THF). Custom XYZ input supported. Downloadable cube files for HOMO, LUMO, and density. Python script generation for reproducible calculations with detailed convergence logs.

### Key Advantages
100% Pure Python including molecular integrals. Numba JIT acceleration for near-C++ performance. GPU support via CuPy. Near-quadratic scaling (~O(N²·⁰⁵)). High accuracy matching PySCF (<10⁻⁷ Ha). Cross-platform compatibility. Easy pip installation.

## Quick Start

### Option 1: Use Online (Recommended)
Simply visit any of the live demo URLs above - no installation required.

### Option 2: Run Locally

Clone the repository:
```bash
git clone https://github.com/manassharma07/pyfock-gui.git
cd pyfock-gui
```

Install dependencies:
```bash
# Install LibXC (required by PyFock)
# For Python < 3.10:
sudo apt-get install libxc-dev     # Ubuntu/Debian
pip install pylibxc2

# For Python >= 3.10 (recommended):
conda install -c conda-forge pylibxc -y

# Install PyFock and dependencies
pip install pyfock streamlit py3Dmol pyscf ase pandas

# Optional: GPU support
pip install cupy-cuda12x   # adjust for your CUDA version
```

Run the app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Basic Workflow

Select a molecule from 15+ examples or paste your own XYZ coordinates. Configure calculation parameters: basis set, XC functional, convergence criteria, and optionally enable PySCF comparison. Adjust visualization settings including cube file resolution, isovalue, and opacity. Run the calculation and monitor progress through status updates. Explore results including energy components, HOMO-LUMO gap, orbital energies, and interactive 3D visualizations. Download cube files and generated Python scripts.

### Example Calculation

```python
# The GUI generates ready-to-run Python scripts
# Example for water molecule with PBE/sto-3g

from pyfock import Basis, Mol, DFT, Utils

# Initialize molecule
mol = Mol(coordfile='water.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='sto-3g')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

# Run DFT
dftObj = DFT(mol, basis, auxbasis, xc=[101, 130])  # PBE
energy, dmat = dftObj.scf()

# Generate cube files
Utils.write_orbital_cube(mol, basis, dftObj.mo_coefficients[:, homo_idx], 'HOMO.cube')
```

## Important Notes

### JIT Compilation
PyFock uses Numba JIT compilation for acceleration. The **first calculation will be slower** as functions are compiled. **Subsequent runs will be significantly faster** - this is expected behavior and a key advantage of JIT compilation.

### System Limits
Cloud version is limited to ~120 basis functions due to computational constraints. Local version can handle much larger systems (~10,000 basis functions). For large molecules, use smaller basis sets (sto-3g) or run locally.

### Recommended Settings
Quick tests: sto-3g basis with small molecules (water, benzene). Production runs: 6-31G or def2-SVP basis. High accuracy: cc-pVDZ or def2-TZVP basis (use locally for large systems).

## Example Molecules

The GUI includes 15+ pre-configured molecules:

| Molecule | Atoms | Description |
|----------|-------|-------------|
| Water | 3 | Quick test system |
| Benzene | 12 | Aromatic ring |
| Acetone | 10 | Carbonyl group |
| Pyrrole | 10 | Heterocycle |
| THF | 13 | Cyclic ether |
| CO₂ | 3 | Linear molecule |
| H₂O₂ | 4 | Peroxide linkage |

## Performance Highlights

### CPU Performance
Up to 2× faster than PySCF with strong scaling up to 32 cores. Near-quadratic scaling (~O(N²·⁰⁵)) with basis functions.

### GPU Acceleration
Up to 14× speedup vs 4-core CPU. Single A100 GPU handles 4000+ basis functions. Consumer GPUs (RTX series) supported.

## Technical Details

### Supported Methods
Kohn-Sham density functional theory with density fitting (resolution of identity approximation) and DIIS-accelerated SCF convergence.

### XC Functionals
LDA: SVWN5 (Slater + VWN5 correlation)  
GGA: PBE, BLYP, BP86

### Basis Sets
STO-3G, STO-6G, 3-21G, 4-31G, 6-31G, 6-31+G, 6-31++G, cc-pVDZ, def2-SVP, def2-TZVP

## Technology Stack

Backend powered by [PyFock](https://github.com/manassharma07/PyFock) for pure Python DFT calculations. Frontend built with [Streamlit](https://streamlit.io/). Visualization using [py3Dmol](https://3dmol.csb.pitt.edu/). Optional comparison with [PySCF](https://pyscf.org/). Structure handling via [ASE](https://wiki.fysik.dtu.dk/ase/).

## Documentation

PyFock Documentation: [https://pyfock-docs.bragitoff.com](https://pyfock-docs.bragitoff.com)  
PyFock GitHub: [https://github.com/manassharma07/PyFock](https://github.com/manassharma07/PyFock)  
PyPI Package: [https://pypi.org/project/pyfock/](https://pypi.org/project/pyfock/)

## Contributing

Contributions are welcome. Please submit a Pull Request or open an issue to discuss major changes.

## Citation

If you use PyFock or PyFock GUI in your research, please cite:

```bibtex
@software{pyfock2025,
  author = {Sharma, Manas},
  title = {PyFock: A Pure Python Gaussian Basis DFT Code for CPU and GPU},
  year = {2025},
  url = {https://github.com/manassharma07/PyFock},
  journal={[Journal Name]},
  note={Manuscript in preparation}
}
```

Paper coming soon on arXiv.

## Author

**Manas Sharma**  
Website: [bragitoff.com](https://bragitoff.com)  
LinkedIn: [manassharma07](https://www.linkedin.com/in/manassharma07)  
Contact: Via GitHub issues

## Support

If you find this project useful:
- Star the [PyFock](https://github.com/manassharma07/PyFock) repository
- Star this [PyFock GUI](https://github.com/manassharma07/pyfock-gui) repository
- Share with colleagues and students
- Report bugs and request features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the PyFock development team, Streamlit community, PySCF developers, and all contributors and users.

---



**Made with care by PhysWhiz**

*Pure Python • Numba JIT • GPU Ready*

[Try Now](https://pyfock.streamlit.app) | [Documentation](https://pyfock-docs.bragitoff.com) | [Issues](https://github.com/manassharma07/pyfock-gui/issues)

