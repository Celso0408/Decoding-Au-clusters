# Decoding High-Dimensional Landscapes

A modernized, rigorously verified Python framework designed for non-linear dimensionality reduction (NLDR) and thermodynamic analysis of complex high-dimensional systems.

This repository centralizes industry-standard dimensionality reduction algorithms (UMAP, FIt-SNE, Sketch-map) and MBAR thermodynamic reweighting into a single, domain-agnostic ecosystem. It is specifically engineered to handle massive datasets with high mathematical fidelity and computational efficiency.

---

## 🚀 Core Features

*   **Unified API (`DimRedPy`)**: A clean, single-entrypoint Python framework wrapping `openTSNE`, `umap-learn`, `pymbar`, and a custom pure-Python/PyTorch re-implementation of the `Sketch-map` algorithm.
*   **Massive Scalability & GPU Acceleration**: Seamless fallback to PyTorch and NVIDIA `cuML` for processing massive datasets (millions of points) where legacy implementations fail due to memory constraints.
*   **Rigorous Mathematical Parity**: The framework is heavily tested against upstream legacy C++ frameworks using Procrustes Disparity, Mean Squared Error, and Pearson Correlation to guarantee $R \approx 1.0$ mathematical equivalence.
*   **Out-of-Sample Projection**: Implements memory-safe subsetting and batch projection strategies for robust thermodynamic landscape construction without causing Out-Of-Memory (OOM) errors.

---

## 🏗 Project Structure

The repository is modularized into four main components:

### 1. `dimredpy/` (The Core Framework)
The standalone python package containing the unified implementations.
*   **Sketch-map**: Pure-Python port of the Sketchmap algorithm using SciPy/PyTorch optimization.
*   **FIt-SNE**: Wrappers for `openTSNE` and GPU `cuML` with automatic parameter resolution.
*   **UMAP**: Wrapped interface using `umap-learn` and GPU `cuML`.
*   **MBAR**: Integration with `pymbar` for projecting free-energy surfaces.

### 2. `Au13_study/` (Scientific Application)
The primary real-world application of the framework. It demonstrates the complete end-to-end global thermodynamic projection and MBAR weighting on a massive 2.16 Million point trajectory of Au13 clusters. It implements the optimal subsets, distance metrics, and visualization plots.

### 3. `dimredpy_vs_others/` (Parity Validation Suite)
A comprehensive suite of scripts proving the statistical equivalence of `dimredpy` against legacy tools. It runs isolated, interleaved benchmarks measuring:
*   Projection Coordinates MSE
*   Procrustes Disparity
*   Pearson Correlation

### 4. `reference_frameworks/`
A static snapshot of the legacy upstream C++ repositories (e.g., the original Sketch-map CLI tools) used purely as baselines for the parity validation suite.

---

## 🛠 Installation & Setup

We highly recommend running this framework in an environment with GPU support (CUDA 12+) to take full advantage of the `cuML` backends.

1.  Clone the repository:
    ```bash
    git clone https://github.com/Celso0408/Decoding-Au-clusters.git
    cd Decoding-Au-clusters
    ```

2.  Install the core framework and dependencies in editable mode:
    ```bash
    pip install -e .
    ```

3.  *(Optional)* Install cuML for GPU acceleration (Linux/WSL):
    ```bash
    pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
    ```

---

## 🔬 Execution Examples

### Running the Global Thermodynamic Pipeline (Au13)
To execute the massive dataset projection, grid search, and optimal MBAR plotting:
```bash
python Au13_study/all_au13.py
```

### Running the Validation Suite
To independently verify the mathematical parity of the framework against legacy tools:
```bash
# Verify Sketch-map Projection Parity
python dimredpy_vs_others/dimredpy_vs_sketchmap/verify_parity_sketchmap_projection.py

# Verify FIt-SNE Parity
python dimredpy_vs_others/dimredpy_vs_fitsne/verify_parity_fitsne.py
```

---

## 📚 References
*   **Sketch-map**
*   **FIt-SNE**
*   **UMAP**
*   **MBAR**
