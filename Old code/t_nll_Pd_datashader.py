#!/usr/bin/env python3
# Datashader raster plot (CPU) — continuous Spectral_r hue

import os, duckdb, datashader as ds, datashader.transfer_functions as tf
from datashader.utils import export_image
import pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as cm

# ---------- parameters -----------------------------------------------------
PARQUET_PTH = r"D:/NLL_matrices/1.4B_supermatrix.parquet"
T_MAX       = 6000
PD_MIN, PD_MAX = -3952, 2047
Y_MAX       = 10
TARGET_PTS  = 5_000_000
OUT_PNG     = "t_nll_Pd_datashader.png"

# ---------- DuckDB: sample exactly TARGET_PTS rows -------------------------
con = duckdb.connect()
con.execute(f"PRAGMA threads = {max(1, os.cpu_count()-2)}")

SQL = f"""
WITH base AS (
  SELECT t, k+1 AS P_t, nll
  FROM read_parquet('{PARQUET_PTH}')
  WHERE t <= {T_MAX} AND nll <= {Y_MAX}
),
calc AS (
  SELECT t, nll, P_t - t AS P_d
  FROM base
  WHERE P_t - t BETWEEN {PD_MIN} AND {PD_MAX}
)
SELECT *
FROM calc
USING SAMPLE {TARGET_PTS} ROWS
"""

print("[INFO] streaming + sampling via DuckDB …")
df = con.execute(SQL).fetch_df()
print(f"[INFO] retrieved {len(df):,} rows")

from matplotlib.colors import to_hex          # add once at the top

# ------------------------------------------------------------------
# 1.  Bin P_d into 20 integer categories  --------------------------
# ------------------------------------------------------------------
N_BINS   = 20
BIN_SIZE = (PD_MAX - PD_MIN + 1) / N_BINS
df["P_d_bin"] = ((df["P_d"] - PD_MIN) // BIN_SIZE).astype("category")

# Build a 20-colour list and colour_key dict
from matplotlib.colors import to_hex
palette     = [to_hex(cm.Spectral_r(i/(N_BINS-1))) for i in range(N_BINS)]
color_key   = {i: palette[i] for i in range(N_BINS)}  # id → colour

# ------------------------------------------------------------------
# 2.  Categorical aggregation  (= 3-D xarray cube)  ----------------
# ------------------------------------------------------------------
canvas = ds.Canvas(plot_width=1000, plot_height=500,
                   x_range=(0, T_MAX), y_range=(0, Y_MAX))

agg = canvas.line(
        df, "t", "nll",
        agg=ds.by("P_d_bin", ds.count())      # <-- keep channels separate
      )                                       # resulting shape: (y, x, 20)

# ------------------------------------------------------------------
# 3.  Render each channel with its own colour, then alpha-composite
# ------------------------------------------------------------------
img = tf.stack(agg, color_key=color_key, how='over')   # categorical shading



export_image(img, OUT_PNG[:-4], fmt=".png") # fmt WITH dot
print(f"[INFO] saved → {OUT_PNG}")

# ---------- display in Matplotlib -----------------------------------------
plt.figure(figsize=(10, 5))
plt.imshow(img.to_pil(), origin="lower")
plt.xlabel("t (token index in document)")
plt.ylabel("NLL (nats)")
plt.title("Datashader raster – NLL vs t (continuous signed P_d hue)")
plt.tight_layout()
plt.show()
