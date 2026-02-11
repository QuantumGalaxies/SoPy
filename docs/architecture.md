# Core Concepts and Architecture

## 1. The SoP Object

The central data structure in SoPy is the `SoPy` object. It is designed to represent hyper-dimensional vectors efficiently.

### Internal Structure
It is internally comprised of:
* **Amplitudes:** A list of coefficients.
* **Components List:** A corresponding list of vector components.

This structure allows the object to "reach deep" into the space of quantum-like dimensions. The core philosophy is that "quantum" effectively means "more dimensions," which is necessary to represent physical problems that scale quickly into massive vector spaces.

---

## 2. Data Models and Schemas

### Input Formats
The SoPy data structure supports loading two primary input formats:

1.  **Dictionary of Spatial Data:** `Dict[spaces]` (using the `.transpose()` method).
2.  **List of Gaussians:** Directly initialized with Gaussian parameters in appropriate dimensions.

### Output/Export
When exported, the `SoPy` object returns a **list of spatial indices**: `[space]`. This representation allows for efficient handling of separated dimensional representations within hyper-dimensional vectors.

---

## 3. Computation Management

### Execution Model
The execution model is built on **TensorFlow**, offering a dual-mode approach:

* **Lazy Execution (Graph-based):** Ideal for maximum performance optimization on large, fixed computations.
* **Eager Execution:** Perfect for debugging and dynamic workflow development.

### Memory and Resource Handling
All primary vector processing and decomposition occur **in-memory** for maximum speed.

Resource allocation is automatically handled by the underlying TensorFlow framework, which intelligently manages compute placement across available devices (CPU, GPU, etc.). This ensures that even when reducing a massive, complex vector to a simpler set via decomposition, the process remains performant.

---

## 4. Key Terminology

### Sums of Products (SoP)
The SoP method is a decomposition technique originating from the work of Beylkin and Molhenkamp (2005) on the separation of dimensions for physics applications.

**Comparison with Matrix Product States (MPS):**
While SoP shares fundamental similarities with MPS (a common quantum reduction method), they differ significantly in scalability:

* **MPS:** Primarily effective in 1D spaces. Requires pre-defined first quantization basis sets.
* **SoP:** Successfully published in up to **12D spaces** using coordinate separation and dense particle-positions per coordinate (PCCP 2022).
    * *Key Advantage:* Methods leveraging SoP are designed to **solve for these basis sets** (including digital orbitals and geminals), rather than requiring them as inputs.

### Fibonacci Decomposition

This is the advanced decomposition scheme unique to SoPy.

**How it works:**
Instead of relying on iterative self-reference (which can be unstable), the Fibonacci method builds a low-rank approximation from the ground up. It composes blocks of like vectors into a canonical **Rank-1 form**, then combines all blocks for a larger training SoP.

**Why use it?**
This technique is designed to enhance the **speed and numerical stability** of complex SoP decompositions.

---

## 5. Extending SoPy

### Custom Modules
Users can build custom modules or plugins to extend the library's functionality.

### Serialization for Custom Containers
If you build custom `[space]` container objects, you must implement the required serialization mechanism. This ensures that your custom data structures can be saved and persisted across sessions.