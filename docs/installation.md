# Getting Started

## 1. Prerequisites

**Python Version:**
SoPy is currently verified for **Python 3.11.14**.
> **Note:** This specific version is recommended due to compatibility requirements with TensorFlow on macOS installations.

## 2. Basic Installation

You can install SoPy directly from PyPI or from the source repository.

### Option A: PyPI (Recommended)
```bash
pip install sopy-quantum
```
### Option B: GitHub Source

If you need the absolute latest development version:

pip install git+[https://github.com/quantumgalaxies/SoPy](https://github.com/quantumgalaxies/SoPy)

## 3. Advanced Features (Optional)
To enable advanced features like PySCF integration (accessed via the sopy.pscf.ext submodule), you must install the optional dependencies using the "extras" syntax:
```bash
pip install sopy-quantum[pyscf]
```

## 4. Quick Start Script
Once installed, you can run a basic sanity check to ensure the library is working correctly.

Create a file named test_sopy.py:
```python
import sopy
print(f"SoPy successfully imported. Version: {sopy.__version__}")
```

Run it with:
```bash
python test_sopy.py
```

