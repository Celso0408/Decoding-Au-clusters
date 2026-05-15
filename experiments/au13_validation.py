"""
Au13 RE-BOMD → NLDR Pipeline Validation
========================================
Reproduces the core computational pipeline from:
    "Decoding the Configurational Landscapes of Au13 Nanoclusters"

Dataset
-------
The coordination histogram file `full_trajectory.xyz.CN_for_SketchMap` contains
2,160,000 rows × 13 columns (fraction of atoms with coordination number 0–12).
The three charge states are laid out sequentially:
    rows       0 –  719999 : cationic  Au13(+1)
    rows  720000 – 1439999 : neutral   Au13(0)
    rows 1440000 – 2159999 : anionic   Au13(-1)

Paper parameters (Section 2.4.1)
---------------------------------
Sketch-map : σ = 1.2, A = 10.5, B = 1.0, a = 1.0, b = 1.0
FIt-SNE    : perplexity = 250, max_inter = 1000, early_exag_coeff = 1000
UMAP       : metric = "mahalanobis", n_neighbors = 50, min_dist = 0.001

Structural thresholds (Section 2.5)
------------------------------------
2D   : ECN ≤ 3.75,  Rg ≥ 3.75 Å
quasi-2D : 3.75 < ECN < 4.50,  3.10 ≤ Rg ≤ 3.40 Å
3D   : ECN ≥ 4.25,  Rg ≤ 3.00 Å
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TEMPERATURES = [
    250.00, 281.00, 312.00, 351.00, 390.00, 440.00,
    487.00, 548.00, 608.00, 684.00, 760.00, 840.00,
]  # 12 replicas per initial structure

N_ATOMS      = 13
N_REPLICAS   = 12
N_STRUCTURES = 3        # 2D, 3D, ICO initial structures
N_PER_CHARGE = 720_000  # samples per charge state
N_TOTAL      = 2_160_000

# Paper sketch-map hyperparameters
SM_SIGMA = 1.2
SM_A_HD  = 10.5
SM_B_HD  = 1.0
SM_A_LD  = 1.0
SM_B_LD  = 1.0

# Paper FIt-SNE hyperparameters
TSNE_PERPLEXITY     = 250
TSNE_MAX_INTER      = 1000
TSNE_EARLY_EXAG     = 1000

# Paper UMAP hyperparameters
UMAP_METRIC     = "mahalanobis"
UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST    = 0.001

# Structural classification thresholds (Section 2.5)
ECN_2D_MAX   = 3.75
ECN_3D_MIN   = 4.25
RG_2D_MIN    = 3.75   # Å
RG_3D_MAX    = 3.00   # Å

# Subsampling factor for charge-coloured overlay (1 dot per 4000 MD steps)
SUBSAMPLE_FACTOR = 4000


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_coord_histograms(path: str | Path, verbose: bool = True) -> np.ndarray:
    """
    Load the 2,160,000 × 13 coordination histogram file.

    The file stores raw integer coordination counts; this function normalises
    each row to fractions (sum = 1) as described in the paper (Section 4.1):
        "the i-th bin … corresponds to the fraction of atoms in the cluster
         having coordination number i"

    Parameters
    ----------
    path    : path to `full_trajectory.xyz.CN_for_SketchMap` (text or .npy).
    verbose : print progress.

    Returns
    -------
    (2_160_000, 13) float64 array, rows normalised to sum 1.
    """
    path = Path(path)
    if verbose:
        print(f"Loading coordination histograms from {path.name} …")

    if path.suffix == ".npy":
        data = np.load(path)
    else:
        data = np.loadtxt(path)

    if data.shape[1] != N_ATOMS:
        raise ValueError(
            f"Expected 13 columns (coordination bins 0–12), got {data.shape[1]}"
        )

    # Normalise raw counts to fractions (paper Eq. / Section 4.1)
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)   # guard against zero rows
    data = data / row_sums

    if verbose:
        print(f"  Loaded {data.shape[0]:,} configurations  ({data.shape[1]}-dim descriptors)")
    return data


def split_by_charge(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split the full 2.16 M array into cationic / neutral / anionic blocks.

    Layout (paper Section 4.1):
        cationic :      0 –  719 999
        neutral  :  720 000 – 1 439 999
        anionic  : 1 440 000 – 2 159 999
    """
    assert data.shape[0] == N_TOTAL, \
        f"Expected {N_TOTAL} rows, got {data.shape[0]}"
    return {
        "cationic": data[0 : N_PER_CHARGE],
        "neutral":  data[N_PER_CHARGE : 2 * N_PER_CHARGE],
        "anionic":  data[2 * N_PER_CHARGE :],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sketch-map pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_sketchmap(
    all_data: np.ndarray,
    n_landmarks: int = 500,
    preopt_steps: int = 2000,
    grid: Tuple = (1.5, 21, 201),
    global_steps: int = 500,
    results_dir: Optional[Path] = None,
    use_gpu: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete sketch-map pipeline on the pooled dataset.

    Strategy (paper Section 4.2):
    1. Select landmarks by MinMax (Farthest Point Sampling) from the pooled,
       temporally decorrelated data.
    2. Initialise from classical MDS.
    3. Optimise sketch-map stress with CG + global grid refinement.
    4. Project all 2.16 M samples out-of-sample.

    Parameters
    ----------
    all_data      : (N_TOTAL, 13) normalised coordination histograms.
    n_landmarks   : number of sketch-map landmarks.
    preopt_steps  : CG preoptimisation steps.
    grid          : (gwidth, g1, g2) for global grid scan.
    global_steps  : CG steps in global refinement phase.
    results_dir   : directory to cache intermediate results (.npy/.npz).
    verbose       : print progress.

    Returns
    -------
    dict with keys:
        "landmarks_hd"  : (n_landmarks, 13) landmark descriptors
        "landmarks_ld"  : (n_landmarks, 2)  sketch-map coordinates
        "lm_weights"    : (n_landmarks,)    Voronoi weights
        "embedding_all" : (N_TOTAL, 2)      full projected trajectory
    """
    from dimredpy.sketchmap          import select_landmarks, sketch_map, classical_mds, project_out_of_sample

    cache = {}
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        lm_cache   = results_dir / "sm_landmarks.npz"
        sm_cache   = results_dir / "sm_landmark_coords.npy"
        proj_cache = results_dir / "sm_full_embedding.npy"

        if lm_cache.exists():
            if verbose: print("  [cache] Loading landmarks …")
            d = np.load(lm_cache)
            cache["landmarks_hd"] = d["landmarks"]
            cache["lm_weights"]   = d["weights"]
        if sm_cache.exists():
            if verbose: print("  [cache] Loading landmark embeddings …")
            cache["landmarks_ld"] = np.load(sm_cache)
        if proj_cache.exists():
            if verbose: print("  [cache] Loading full embedding …")
            cache["embedding_all"] = np.load(proj_cache)

        if len(cache) == 4:
            return cache

    # ── Step 1: Landmark selection ──────────────────────────────────────────
    if "landmarks_hd" not in cache:
        # Decorrelate by striding (remove temporal autocorrelation)
        stride_data = all_data[::20]
        if verbose:
            print(f"  Selecting {n_landmarks} landmarks from "
                  f"{stride_data.shape[0]:,} decorrelated frames …")
        lm = select_landmarks(stride_data, n_landmarks=n_landmarks, mode="minmax")
        cache["landmarks_hd"] = lm["landmarks"]
        cache["lm_weights"]   = lm["weights"]
        if results_dir is not None:
            np.savez(lm_cache,
                     landmarks=lm["landmarks"], weights=lm["weights"])

    landmarks_hd = cache["landmarks_hd"]
    lm_weights   = cache["lm_weights"]

    # ── Step 2: MDS initialisation ──────────────────────────────────────────
    if verbose: print("  Initialising from classical MDS …")
    init_pos = classical_mds(landmarks_hd, n_components=2)["embedding"]

    # ── Step 3: Sketch-map optimisation ────────────────────────────────────
    if "landmarks_ld" not in cache:
        if verbose: print("  Running sketch-map CG + global grid refinement …")
        sm = sketch_map(
            landmarks_hd,
            n_components=2,
            weights=lm_weights,
            init=init_pos,
            fun_hd=(SM_SIGMA, SM_A_HD, SM_B_HD),
            fun_ld=(SM_SIGMA, SM_A_LD, SM_B_LD),
            preopt_steps=preopt_steps,
            grid=grid,
            global_steps=global_steps,
            verbose=verbose,
        )
        cache["landmarks_ld"] = sm["embedding"]
        if results_dir is not None:
            np.save(sm_cache, sm["embedding"])
        if verbose:
            print(f"  Final χ² stress: {sm['stress']:.6f}")

    landmarks_ld = cache["landmarks_ld"]

    # ── Step 4: Project all samples ─────────────────────────────────────────
    if "embedding_all" not in cache:
        if verbose: print(f"  Projecting all {all_data.shape[0]:,} samples …")
        proj = project_out_of_sample(
            all_data,
            landmarks_hd,
            landmarks_ld,
            weights=lm_weights,
            fun_hd=(SM_SIGMA, SM_A_HD, SM_B_HD),
            fun_ld=(SM_SIGMA, SM_A_LD, SM_B_LD),
            grid=grid,
            cg_steps=3,
            use_gpu=use_gpu,
            verbose=verbose,
        )
        cache["embedding_all"] = proj["embedding"]
        if results_dir is not None:
            np.save(proj_cache, proj["embedding"])

    return cache


# ──────────────────────────────────────────────────────────────────────────────
# FIt-SNE pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_fitsne(
    all_data: np.ndarray,
    results_dir: Optional[Path] = None,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run FIt-SNE on the full dataset with paper hyperparameters.

    Paper: perplexity = 250, max_inter = 1000, early_exag_coeff = 1000.
    Uses openTSNE on CPU (or cuml on GPU if available).

    Returns
    -------
    (N_TOTAL, 2) embedding coordinates.
    """
    from dimredpy.fitsne import fit_sne

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_path = results_dir / "fitsne_embedding.npy"
        if cache_path.exists():
            if verbose: print("  [cache] Loading FIt-SNE embedding …")
            return np.load(cache_path)

    if verbose:
        print(f"  Running FIt-SNE (perplexity={TSNE_PERPLEXITY}, "
              f"early_exag={TSNE_EARLY_EXAG}) …")

    emb = fit_sne(
        all_data,
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=1000,
        early_exaggeration=TSNE_EARLY_EXAG,
        early_exaggeration_iter=250,
        min_num_intervals=TSNE_MAX_INTER,
        learning_rate="auto",
        use_gpu=use_gpu,
        verbose=verbose,
    )

    if results_dir is not None:
        np.save(cache_path, emb)
    return np.asarray(emb)


# ──────────────────────────────────────────────────────────────────────────────
# UMAP pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_umap(
    all_data: np.ndarray,
    results_dir: Optional[Path] = None,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run UMAP on the full dataset with paper hyperparameters.

    Paper: metric = "mahalanobis", n_neighbors = 50, min_dist = 0.001.

    Returns
    -------
    (N_TOTAL, 2) embedding coordinates.
    """
    from dimredpy.umap_embed import umap_embed

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_path = results_dir / "umap_embedding.npy"
        if cache_path.exists():
            if verbose: print("  [cache] Loading UMAP embedding …")
            return np.load(cache_path)

    if verbose:
        print(f"  Running UMAP (metric={UMAP_METRIC}, "
              f"n_neighbors={UMAP_N_NEIGHBORS}, "
              f"min_dist={UMAP_MIN_DIST}) …")

    emb = umap_embed(
        all_data,
        n_components=2,
        metric=UMAP_METRIC,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        use_gpu=use_gpu,
        verbose=verbose,
    )

    if results_dir is not None:
        np.save(cache_path, emb)
    return np.asarray(emb)


# ──────────────────────────────────────────────────────────────────────────────
# Structural descriptors
# ──────────────────────────────────────────────────────────────────────────────

def compute_ecn_from_histogram(coord_hist: np.ndarray) -> np.ndarray:
    """
    Compute the Effective Coordination Number proxy from a coord histogram.

    Because the file stores the fraction of atoms at each coordination level,
    a simple weighted mean gives the mean coordination number:
        ECN_proxy = Σ_i  i * hist[i]   where i ∈ 0..12

    This matches the ECN values described in the paper (range ~3–6).

    Parameters
    ----------
    coord_hist : (N, 13) normalised coordination histograms.

    Returns
    -------
    (N,) ECN proxy values.
    """
    i_values = np.arange(N_ATOMS, dtype=float)
    return coord_hist @ i_values   # dot product: sum_i(i * hist_i)


def compute_rg_from_histogram(coord_hist: np.ndarray) -> np.ndarray:
    """
    Estimate Radius of Gyration (Rg) proxy from the coordination histogram.

    Lower coordination → more open 2D structure → larger Rg.
    We use: Rg_proxy = 3.75 - 0.5 * (ECN - 3.75)  (linear interpolation from
    paper thresholds: ECN=3.75 → Rg=3.75, ECN=5.5 → Rg=2.65).
    This gives a plausible Rg for symbol-size mapping.

    For real Rg you need the 3D atomic positions.
    """
    ecn = compute_ecn_from_histogram(coord_hist)
    # Linear map: ECN=3.75 → Rg=3.75 Å, ECN=5.5 → Rg=2.65 Å
    rg = 3.75 - (ecn - 3.75) * (3.75 - 2.65) / (5.5 - 3.75)
    return np.clip(rg, 2.65, 4.05)


def classify_morphology(ecn: np.ndarray) -> np.ndarray:
    """
    Classify each frame as '3D', 'quasi-2D', or '2D' using paper thresholds.

    Returns string array of labels.
    """
    labels = np.full(len(ecn), "quasi-2D", dtype=object)
    labels[ecn >= ECN_3D_MIN]  = "3D"
    labels[ecn <= ECN_2D_MAX]  = "2D"
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Full validation pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    data_path: str | Path,
    results_dir: str | Path = "./results",
    run_sketchmap_flag: bool = True,
    run_fitsne_flag:    bool = True,
    run_umap_flag:      bool = True,
    n_landmarks: int = 500,
    use_gpu: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end pipeline: load → NLDR × 3 → structural descriptors.

    Parameters
    ----------
    data_path     : path to coordination histogram file.
    results_dir   : directory for caching intermediate results.
    run_*_flag    : toggle each NLDR method.
    n_landmarks   : number of sketch-map landmarks.
    verbose       : print progress.

    Returns
    -------
    dict with keys: "data", "charges", "embeddings", "ecn", "rg".
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    data    = load_coord_histograms(data_path, verbose=verbose)
    charges = split_by_charge(data)

    embeddings: Dict[str, Dict] = {}

    # ── Sketch-map ──────────────────────────────────────────────────────────
    if run_sketchmap_flag:
        if verbose: print("\n[Sketch-map]")
        sm_out = run_sketchmap(data, n_landmarks=n_landmarks,
                               results_dir=results_dir / "sketchmap",
                               use_gpu=use_gpu,
                               verbose=verbose)
        embeddings["sketchmap"] = {
            k: {lbl: sm_out["embedding_all"][i * N_PER_CHARGE:(i+1) * N_PER_CHARGE]
                for i, lbl in enumerate(["cationic", "neutral", "anionic"])}
            for k in ["embedding_per_charge"]
        }
        embeddings["sketchmap"]["full"] = sm_out["embedding_all"]
        embeddings["sketchmap"]["landmarks_hd"] = sm_out["landmarks_hd"]
        embeddings["sketchmap"]["landmarks_ld"] = sm_out["landmarks_ld"]
        embeddings["sketchmap"]["lm_weights"]   = sm_out["lm_weights"]

    # ── FIt-SNE ─────────────────────────────────────────────────────────────
    if run_fitsne_flag:
        if verbose: print("\n[FIt-SNE]")
        tsne_emb = run_fitsne(data, results_dir=results_dir / "fitsne",
                              use_gpu=use_gpu,
                              verbose=verbose)
        embeddings["fitsne"] = {
            "full": tsne_emb,
            "cationic": tsne_emb[0:N_PER_CHARGE],
            "neutral":  tsne_emb[N_PER_CHARGE:2*N_PER_CHARGE],
            "anionic":  tsne_emb[2*N_PER_CHARGE:],
        }

    # ── UMAP ────────────────────────────────────────────────────────────────
    if run_umap_flag:
        if verbose: print("\n[UMAP]")
        umap_emb = run_umap(data, results_dir=results_dir / "umap",
                            use_gpu=use_gpu,
                            verbose=verbose)
        embeddings["umap"] = {
            "full": umap_emb,
            "cationic": umap_emb[0:N_PER_CHARGE],
            "neutral":  umap_emb[N_PER_CHARGE:2*N_PER_CHARGE],
            "anionic":  umap_emb[2*N_PER_CHARGE:],
        }

    # ── Structural descriptors ───────────────────────────────────────────────
    if verbose: print("\n[Structural descriptors]")
    ecn = {k: compute_ecn_from_histogram(v) for k, v in charges.items()}
    rg  = {k: compute_rg_from_histogram(v)  for k, v in charges.items()}
    ecn["all"] = compute_ecn_from_histogram(data)
    rg["all"]  = compute_rg_from_histogram(data)

    if verbose: print("\nPipeline complete.")

    return {
        "data":       data,
        "charges":    charges,
        "embeddings": embeddings,
        "ecn":        ecn,
        "rg":         rg,
    }


if __name__ == "__main__":
    import sys
    
    # Handle notebook execution where __file__ is not defined
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd() / "experiments"
        if not base_dir.exists():
            base_dir = Path.cwd()
            
    data_file = base_dir / "full_trajectory.xyz.CN_for_SketchMap"
    results_dir = base_dir / "results"
    
    # Check for GPU (optional: user can force CPU by changing use_gpu=False below)
    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            use_gpu = False
    except ImportError:
        use_gpu = False

    if not data_file.exists():
        print(f"Error: Dataset not found at {data_file}")
        print("Please ensure full_trajectory.xyz.CN_for_SketchMap is in the experiments folder.")
        sys.exit(1)
        
    print("="*60)
    print("  DIMREDPY: Au13 CONFIGURATIONAL LANDSCAPE PIPELINE")
    print(f"  Mode: {'GPU' if use_gpu else 'CPU (Slow Mode)'}")
    print("="*60)
    
    results = run_full_pipeline(
        data_path=data_file,
        results_dir=results_dir,
        run_sketchmap_flag=True,
        run_fitsne_flag=True,
        run_umap_flag=True,
        n_landmarks=500,
        use_gpu=use_gpu,
        verbose=True
    )
    
    print("\nSummary of results:")
    for method, emb in results["embeddings"].items():
        print(f"  - {method}: {emb['full'].shape} embedding generated.")
    
    print(f"\nAll results saved to: {results_dir}")
