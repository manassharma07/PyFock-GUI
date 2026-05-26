# PyFock GUI - Interactive DFT Calculations in Your Browser

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pyfock.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyFock](https://img.shields.io/badge/PyFock-Latest-orange.svg)](https://github.com/manassharma07/PyFock)

A modern, interactive web interface for [PyFock](https://github.com/manassharma07/PyFock), a pure Python Gaussian-basis DFT code with Numba JIT acceleration, density fitting, PySCF validation, and optional CUDA acceleration through CuPy.

![PyFock GUI Banner](https://raw.githubusercontent.com/manassharma07/PyFock/main/logo_crysx_pyfock.png)

## Live Demo

Try PyFock GUI instantly without local installation:

- Primary: [https://pyfock.streamlit.app](https://pyfock.streamlit.app)
- Alternative: [https://pyfock-gui.bragitoff.com](https://pyfock-gui.bragitoff.com)
- HuggingFace: [https://manassharma07-pyfock-gui.hf.space/](https://manassharma07-pyfock-gui.hf.space/)

## Features

### Computational Capabilities

Run Kohn-Sham DFT and HF calculations with density fitting, DIIS convergence, selectable CAO/SAO basis representation, built-in PySCF grid generation for XC terms, and optional PySCF energy comparison. For HF, PySCF comparison uses RIHF rather than a DFT calculation.

### XC Functionals

The GUI exposes the native PyFock functional list:

- HF
- LDA
- PBE, PBESOL, RPBE, PW91
- BP86, BLYP
- R2SCAN, TPSS, M06L, TASK

LibXC is not required for the GUI functional list above.

### Visualization

Interactive 3D structure viewing with py3Dmol, HOMO/LUMO cube generation, electron density cubes, adjustable isosurfaces and opacity, and on-demand visualization of any molecular orbital.

### Input/Output

Choose from preconfigured example molecules or paste custom XYZ coordinates. The GUI can download HOMO, LUMO, density cube files, and a generated Python script that reproduces the PyFock calculation.

### Key Advantages

PyFock is 100% pure Python, including molecular integral evaluation, with Numba JIT acceleration, density fitting with Cauchy-Schwarz screening, near-quadratic scaling for Coulomb terms, PySCF-level numerical accuracy, and optional GPU support.

## Quick Start

### Option 1: Use Online

Visit one of the live demo URLs above. No local setup is required.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/manassharma07/PyFock-GUI.git
cd PyFock-GUI
```

Install dependencies:

```bash
pip install pyfock streamlit py3Dmol pyscf ase pandas plotly

# Optional: GPU support, choose the package matching your CUDA version
pip install cupy-cuda12x
```

Run the app:

```bash
streamlit run Home.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage Guide

### Basic Workflow

Select an example molecule or paste custom XYZ coordinates. Choose the basis set, auxiliary basis, XC functional, convergence settings, and CAO or SAO representation. For DFT functionals, the GUI can use PyFock's built-in PySCF grid path for the XC term. Enable PySCF comparison when you want an RIHF or KS-DFT reference energy using the same molecular setup.

After the calculation, inspect total energy, energy components, HOMO-LUMO gap, orbital energies, density matrix, convergence history, molecule/orbital visualizations, cube downloads, and the generated Python input script.

### Example Calculation

```python
from pyfock import Basis, Mol, DFT, Utils
import numpy as np

mol = Mol(coordfile='water.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='sto-3g')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

dftObj = DFT(mol, basis, auxbasis, xc='PBE')
dftObj.conv_crit = 1e-6
dftObj.max_itr = 14
dftObj.sao = False

energy, dmat = dftObj.scf()
print(f"Total Energy: {energy:.8f} Ha")

occupied = np.where(dftObj.mo_occupations > 1e-8)[0]
homo_idx = occupied[-1]
Utils.write_orbital_cube(mol, basis, dftObj.mo_coefficients[:, homo_idx], 'HOMO.cube')
```

## Important Notes

### JIT Compilation

PyFock uses Numba JIT compilation. The first calculation in a fresh Python session can be slower while kernels compile; subsequent calculations are usually much faster.

### System Limits

The cloud GUI limits calculations to roughly 120 basis functions. Local runs can handle much larger systems depending on memory, CPU/GPU hardware, and basis size.

### Recommended Settings

Use `sto-3g` with small molecules for quick tests. Use `6-31G`, `cc-pvDZ`, `def2-SVP`, or `def2-TZVP` locally for larger or more accurate calculations.

## Example Molecules

The GUI includes these preconfigured molecules:

| Molecule | Atoms | Description |
|----------|-------|-------------|
| Water | 3 | Quick test system |
| Acetone | 10 | Carbonyl-containing molecule |
| Tetrahydrofuran | 13 | Cyclic ether |
| Pyrrole | 10 | Aromatic heterocycle |
| Dimethyl ether | 9 | Ether |
| Benzene | 12 | Aromatic ring |
| Carbon dioxide | 3 | Linear molecule |
| Hydrogen peroxide | 4 | Peroxide linkage |
| Formic acid | 5 | Small carboxylic acid |
| Hydrogen sulfide | 3 | Sulfur analogue of water |
| AgCl | 2 | Silver chloride diatomic |
| AuCl | 2 | Gold chloride diatomic |
| Cd dimer | 2 | Cadmium dimer |

Custom XYZ input is also supported.

## Performance Highlights

### CPU Performance

PyFock uses Numba, NumPy, NumExpr, SciPy, and Joblib to achieve efficient pure-Python DFT calculations, with near-quadratic scaling for density-fitted Coulomb terms and strong multicore CPU support.

### GPU Acceleration

PyFock supports CUDA acceleration through CuPy and Numba. Current project materials report up to 24x speedup on an A100 GPU versus a 4-core CPU, with single-GPU calculations demonstrated for systems with thousands of basis functions.

## Technical Details

### Supported Methods

- RHF/RIHF through the HF functional option
- Kohn-Sham DFT with density fitting
- DIIS-accelerated SCF
- CAO and SAO basis representations
- Optional PySCF energy comparison

### Basis Sets

`sto-3g`, `sto-6g`, `3-21G`, `4-31G`, `6-31G`, `6-31+G`, `6-31++G`, `cc-pvDZ`, `def2-SVP`, `def2-TZVP`

The default auxiliary basis in the GUI is `def2-universal-jfit`.

## Technology Stack

Backend powered by [PyFock](https://github.com/manassharma07/PyFock). Frontend built with [Streamlit](https://streamlit.io/). Molecular visualization uses [py3Dmol](https://3dmol.csb.pitt.edu/). Optional comparison and grid generation use [PySCF](https://pyscf.org/). Structure parsing uses [ASE](https://wiki.fysik.dtu.dk/ase/).

## Documentation

PyFock Documentation: [https://pyfock-docs.bragitoff.com](https://pyfock-docs.bragitoff.com)  
PyFock GitHub: [https://github.com/manassharma07/PyFock](https://github.com/manassharma07/PyFock)  
PyPI Package: [https://pypi.org/project/pyfock/](https://pypi.org/project/pyfock/)

## Contributing

Contributions are welcome. Please submit a pull request or open an issue to discuss major changes.

## Citation

If you use PyFock or PyFock GUI in your research, please cite:

```bibtex
@article{sharma2025pyfock,
  title={PyFock: A Just-In-Time Compiled Gaussian Basis DFT Python Code for CPU and GPU Architectures},
  author={Sharma, Manas and Sierka, Marek},
  journal={[Journal Name]},
  year={2026},
  note={Manuscript in preparation}
}
```

## Author

**Manas Sharma**
Website: [manas.bragitoff.com](https://manas.bragitoff.com)
LinkedIn: [manassharma07](https://www.linkedin.com/in/manassharma07)
Contact: Via GitHub issues

## Support

If you find this project useful:

- Star the [PyFock](https://github.com/manassharma07/PyFock) repository
- Star this [PyFock GUI](https://github.com/manassharma07/PyFock-GUI) repository
- Share with colleagues and students
- Report bugs and request features

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Thanks to the PyFock development team, Streamlit community, PySCF developers, and all contributors and users.

---

**Made with care by PhysWhiz**

*Pure Python - Numba JIT - GPU Ready*

[Try Now](https://pyfock.streamlit.app) | [Documentation](https://pyfock-docs.bragitoff.com) | [Issues](https://github.com/manassharma07/PyFock-GUI/issues)
