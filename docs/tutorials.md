# Tutorials and Recipes

This section provides step-by-step guides for common SoPy workflows, from basic decomposition to advanced Fourier analysis.

---

## 1. Decomposing a Complex SoP Vector

This tutorial guides you through loading a dataset, processing it, and generating decomposition metrics.

### Steps
1.  **Initialize the SoPy Object:** Define your lattice sites and dimensions.
2.  **Load Data:** Use `SoPy.transpose()` to ingest your dictionary of spatial data.
3.  **Run Decomposition:** Call the `.decompose()` method.
4.  **Analyze Results:** Check the rank reduction and error metrics.

```python
from sopy.vectors import SoPy

# 1. Define Lattices (Example for a 2D space)
# Define exact sampling points for each dimension
lattices = [[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]]

# 2. Initialize and Load Data
my_vector = SoPy(lattices)

# Assuming 'spatial_data' is a Dict[spaces]
# my_vector.transpose(spatial_data)

# 3. Decompose
# Reduces the rank using default algorithms
decomposed_vector = my_vector.decompose()

# 4. Access Results
print(f"Original Rank: {len(my_vector.amplitudes)}")
print(f"Decomposed Rank: {len(decomposed_vector.amplitudes)}")
```

## 2. Fourier Analysis with exp(ikx)

SoPy allows for complex number domain analysis using the exp(ikx) operator. This is essential for moving between spatial and momentum representations in hyper-dimensional space.

Key Concept

The library integrates the operator in SoP form, allowing you to apply it directly to your vectors without leaving the compressed representation.

```python
# Applying the Fourier Operator
# Assuming 'complex_sop' is your initialized SoPy object containing complex data

# Define Momentum values (k)
ks = [1.0, 2.0, 3.0] 

# Apply the operator via the Operand container
# This transforms the vector into the momentum domain
transformed_vector = complex_sop.Operand.exp_i(ks)
```

## 3. PySCF Integration (3D Digital Solutions)
Leverage PySCF solutions within SoPy's framework to produce visual, three-dimensional models of digital orbitals and geminals.

Prerequisites

Ensure you have installed the optional dependencies:

```python
pip install sopy-quantum[pyscf]
```

Solve with PySCF: Generate your wavefunction or orbital data using standard PySCF methods.

Ingest into SoPy: Use the sopy.pscf.ext submodule to convert the PySCF object into a SoPy vector.

Visualize: Use SoPy's plotting tools to render the 3D representation.

Note: See the example usage notebook in examples/pySCF_wavefunction.ipynb for a complete, runnable script.

# Recipes
Generating Synthetic SoP Data

Need to validate your workflow? You can generate valid, structured test data using the Vector class methods.

Gaussians (Vector.gaussian): Good for testing smooth, continuous data decomposition.

Deltas/Sinc (Vector.delta): Best for testing high-frequency components and resolution limits.

# Exporting and Importing Data

SoPy uses a specialized transposition method for efficient I/O.

Importing: Use .transpose(data). This expects a dictionary where keys are spatial indices.

Exporting: When you export a processed SoPy vector, it returns a list of spatial indices: [space].

Note: Amplitudes are stored under key 0 in the dictionary format.