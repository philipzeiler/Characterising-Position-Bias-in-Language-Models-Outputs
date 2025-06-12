#!/usr/bin/env python3
# Compare two HDF5 NLL matrices and print detailed statistics
# -----------------------------------------------------------

import h5py
import numpy as np

# --- hard-code the two paths -----------------------------------------------
FILE_A = "nll_matrices/doc28318_fp16.h5"
FILE_B = "nll_matrices_timetest/doc28318_fp16.h5"   # ← adjust as needed
DSET   = "nll"                                      # dataset name

# ---------------------------------------------------------------------------


def compare_matrices(path_a: str = FILE_A,
                     path_b: str = FILE_B,
                     dset: str = DSET) -> bool:
    """
    Compare two HDF5 datasets.
      • reports shape, bit-exact equality
      • if not identical, prints mean / mean-abs-diff / variance / max diff
    Returns True if arrays are bit-wise identical, else False.
    """
    # ---- load ----------------------------------------------------------------
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        a = fa[dset][...]
        b = fb[dset][...]

    # ---- shape check ---------------------------------------------------------
    if a.shape != b.shape:
        print(f"[ERROR] shape mismatch: {a.shape} vs {b.shape}")
        return False

    # ---- exact equality ------------------------------------------------------
    equal = np.array_equal(a, b)     # bit-for-bit (NaNs must coincide)
    print("IDENTICAL" if equal else "DIFFERENT")

    # ---- if different, compute detailed stats --------------------------------
    if not equal:
        # promote to float32 for numerical stability
        a32, b32 = a.astype(np.float32), b.astype(np.float32)

        # ignore NaNs that may appear in NLL matrices
        mask     = np.isfinite(a32) & np.isfinite(b32)
        if not mask.any():
            print("[WARN] no finite overlap between matrices")
            return False

        diff        = a32[mask] - b32[mask]
        abs_diff    = np.abs(diff)

        mean_a      = np.mean(a32[mask])
        mean_b      = np.mean(b32[mask])
        mean_diff   = np.mean(diff)
        mean_abs    = np.mean(abs_diff)
        var_diff    = np.var(diff)
        max_diff    = np.max(abs_diff)
        idx_max     = np.unravel_index(np.argmax(abs_diff), a32.shape)

        # ---- print report ----------------------------------------------------
        print("\nDetailed difference report (finite elements only)")
        print("-" * 60)
        print(f"Mean   A : {mean_a:.6f}")
        print(f"Mean   B : {mean_b:.6f}")
        print(f"Mean   difference (A-B) : {mean_diff:.6f}")
        print(f"Mean |difference|       : {mean_abs:.6f}")
        print(f"Variance of difference  : {var_diff:.6e}")
        print(f"Max |difference|        : {max_diff:.6f}  @ index {idx_max}")
        print("-" * 60)

    return equal


# ---- run on import ---------------------------------------------------------
if __name__ == "__main__":
    compare_matrices()
