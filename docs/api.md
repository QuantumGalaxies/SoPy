# API Reference

## Module: `sopy.Vectors`

This module contains the primary data structure, initialization, and all core analytical operations for the SoPy vector space. The fundamental data representation is a list of amplitudes paired with a `components_list`.

The internal code is divided into highly optimized containers for performance.

### Core Class

#### `class SoPy(lattices: List[List[float]])`
The main constructor for the SoP object.

* **lattices**: An explicit list of lattice sites for each dimension. This defines the exact sampling points in the hyper-dimensional space.

#### `transpose(data: Dict[spaces])`
A utility method to load data using the transposed dictionary format.
* **data**: A dictionary where keys represent spatial indices and values represent the data at those indices.
* **Returns**: The `SoPy` object populated with the transposed data.

---

### Internal Vector Containers

These classes construct the fundamental components of the `SoPy` vector.

#### `class Vector` (Real-Valued)
Used for standard Gaussian and spatial representations.

* **`Vector.gaussian(amplitude, positions, sigmas, angular, lattices)`**
    * Adds a Gaussian component to the real-valued SoP vector.
    * **amplitude**: The peak height of the Gaussian.
    * **positions**: Center positions of the Gaussian.
    * **sigmas**: Width/standard deviation.
    * **angular**: Angular momentum quantum numbers (l, s).
    * **lattices**: The spatial grid definitions.

* **`Vector.delta(a, positions, spacings, lattices)`**
    * Adds a Dirac delta function approximation (Sinc function) to the real-valued SoP vector.

* **`Vector.dot(other: Vector) -> float`**
    * Calculates the foundational real-valued dot product between two `Vector` containers.

#### `class Operand` (Complex-Valued)
Used for the $\exp(i k x)$ space and Fourier operations.

* **`Operand.cdot(other: Operand) -> complex`**
    * Calculates the foundational complex-valued dot product between two `Operand` containers.

---

### Analysis and Transformations

These methods perform high-level operations on the `SoPy` vectors.

#### `exp_i(ks) -> SoPy`

A core analytical operation performed on the complex vector space containers (`Operand`) for generating the Fourier Transform in the SoP dimensions.
* **ks**: The momentum/wavevector values.
* **Usage**: `complex_sop.Operand.exp_i(ks)`

#### `decompose() -> SoPy`
The standard method for decomposition. Reduces the rank of the `SoPy` vector using default algorithms.

#### `fibonacci() -> SoPy`
Executes the **Advanced Fibonacci** decomposition method.
* **Description**: This scheme builds up a decomposed vector from the ground up without iterative self-reference. It composes blocks of like vectors into a canonical rank-1 form, then combines all blocks for a larger training SoP.
* **Benefits**: Greatly enhances both the speed and numerical stability of complex decompositions compared to standard iterative methods.

---

## Exceptions

#### `InvalidSchemaError`
Raised when input data does not match the expected schema for `SoPy` vectors or lattices.