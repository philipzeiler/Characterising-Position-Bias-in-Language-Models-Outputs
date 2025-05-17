import h5py, numpy as np

# --- hard-code the two paths -------------------------------------------------
FILE_A = "nll_matrices/doc127302_fp16.h5"
FILE_B = "nll_matrices_2/doc127302_gpu.h5"   # change as needed
DSET   = "nll"                             # dataset name in both files

def matrices_identical(path_a=FILE_A, path_b=FILE_B, dset=DSET) -> bool:
    """Return True iff the two datasets are exactly equal bit-for-bit."""
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        a = fa[dset][...]                 # NumPy ndarray  ✔ h5py docs :contentReference[oaicite:0]{index=0}
        b = fb[dset][...]
    if a.shape != b.shape:
        print(f"Shape mismatch: {a.shape} vs {b.shape}")
        return False
    equal = np.array_equal(a, b)          # exact comparison  ✔ NumPy docs :contentReference[oaicite:1]{index=1}
    print("IDENTICAL" if equal else "DIFFERENT")
    return equal

# run once at import time (remove if you want to call later)
if __name__ == "__main__":
    matrices_identical()
