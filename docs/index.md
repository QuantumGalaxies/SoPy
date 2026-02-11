# SoPy: Quantum Galaxies

**Release 1.6.0**

## 1. Introduction

### What is SoPy?
SoPy is a specialized Python library focused on the advanced decomposition of Sums of Products (SoP) tensors.

Unlike traditional methods (such as Tensorly) that reduce dense, hyper-dimensional vectors to SoP form, SoPy provides novel decomposition algorithms—including **Advanced Fibonacci methods**—to reliably decrease the number of terms in complex SoP summations.

It treats Gaussian-like datasets as inherent SoP vectors within a hyper-dimensional vector space. The library also integrates the $\exp(i k x)$ operator in SoP form to facilitate advanced Fourier workflows.

### Key Features
* **Advanced SoP Decomposition:** Novel algorithms for reducing term counts in complex summations.
* **Hyper-dimensional Vector Modeling:** Treats datasets as vectors in high-dimensional spaces (up to 12D).
* **Fourier Workflow Integration:** Native support for the $\exp(i k x)$ operator.
* **PySCF Integration:** Optional support for digital orbitals and geminals (via `sopy-quantum[pyscf]`).

---

## 2. What's New in Release 1.6.0?

This release introduces native support for **Complex Vectors**, allowing for proper handling and decomposition of data in the complex hyper-dimensional space.

* **Complex Space Handling:** Crucial for leveraging the Fourier $\exp(i k x)$ operator with full fidelity.
* **Stability Improvements:** Various bug fixes and performance stability improvements across the core decomposition and data ingestion modules.

---

## 3. Key Terminology & Concepts

### Sums of Products (SoP)
The SoP method is a decomposition technique originating from the work of Beylkin and Molhenkamp (2005) on the separation of dimensions for physics applications.

**SoP vs. Matrix Product States (MPS):**
While SoP shares fundamental similarities with the common quantum reduction method, Matrix Product States (MPS), they differ significantly in scalability:
* **MPS:** Primarily effective in 1D spaces. Requires pre-defined first quantization basis sets.
* **SoP:** Successfully published in up to **12D spaces** using coordinate separation and dense particle-positions per coordinate (PCCP 2022). Methods leveraging SoP are designed to *solve* for basis sets, including digital orbitals and geminals.

### Fibonacci Decomposition
The "Fibonacci" method is an advanced decomposition scheme included in SoPy.

Instead of relying on iterative self-reference, it builds a low-rank approximation from the ground up by composing blocks of like vectors into a canonical rank-1 form. This technique is designed to enhance the speed and numerical stability of complex SoP decompositions.

---

## 4. Quick Start

### Installation

**Basic Install:**
```bash
pip install sopy-quantum
```