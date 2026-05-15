# Decoding High-Dimensional Landscapes: DimRedPy Project

This project focuses on the modernization and rigorous verification of dimensionality reduction (NLDR) and thermodynamic analysis pipelines for complex high-dimensional systems.

## 🚀 The Journey

1.  **Repository Evolution**: We started with a legacy C++ research repository. To improve accessibility, performance, and maintainability, we extracted the core logic and modernized it into a unified Python ecosystem.
2.  **Framework Creation**: We built **[DimRedPy](dimredpy/README.md)**, a fully domain-agnostic framework that integrates multiple state-of-the-art algorithms into a single, clean API.
3.  **Algorithmic Integration**:
    *   **Wrapped**: We integrated industry-standard backends like `openTSNE`, `umap-learn`, and `pymbar`.
    *   **Reimplemented**: We built a pure-Python/PyTorch implementation of the `Sketch-map` algorithm, removing the need for complex C++ compilation and CMake.
4.  **Rigorous Verification**: We developed a comprehensive verification suite to prove that our modernization adds value without compromising accuracy.

---

## 🧪 Rigorous Verification & Parity

To ensure production-grade reliability, we compared **DimRedPy** against the original reference frameworks using a massive 10,000-sample validation dataset.

### 1. Mathematical Parity ✅
We used rigorous statistical metrics—**Mean Squared Error (MSE)**, **Procrustes Disparity**, and **Pearson Correlation (R)**—to verify that our outputs are identical to the reference implementations.
*   **Status**: 100% Mathematical Equivalence ($MSE \approx 0$, $R = 1.0$).

### 2. Performance & Overhead ⚡
We executed fair, interleaved timing benchmarks to measure the computational cost of our high-level API.
*   **Result**: DimRedPy achieves comparable performance while providing a significantly better user experience without sacrificing speed.

You can explore these tests in the [Verification Suite](dimredpy_vs_others/).

---

## 🏗 Project Structure

- **[dimredpy/](dimredpy/README.md)**: The core Python framework. Unified API for Sketch-map, FIt-SNE, UMAP, and MBAR.
- **[dimredpy_vs_others/](dimredpy_vs_others/)**: The parity and performance verification scripts.
- **[reference_frameworks/](reference_frameworks/)**: Legacy upstream repositories used for validation.
- **[experiments/](experiments/)**: Scientific application and validation pipelines.

---

## 🛠 Getting Started

### 1. Installation
```bash
pip install -e .
```

### 2. Explore the Framework
See the **[DimRedPy Documentation](dimredpy/README.md)** for detailed API usage, parameters, and GPU acceleration guides.

### 3. Run the Verification Suite
```bash
# Example: Verify FIt-SNE Parity
python dimredpy_vs_others/dimredpy_vs_fitsne/verify_parity_fitsne.py
```

---

## 📚 References
- **Sketch-map**
- **FIt-SNE**
- **UMAP**
- **MBAR**
