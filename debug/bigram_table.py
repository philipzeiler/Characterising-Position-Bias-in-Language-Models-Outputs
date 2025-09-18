#!/usr/bin/env python3
"""
Figure-style table: Top starting bigrams in the validation set
Outputs a vector PDF that matches your plot styling.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

# ------------------------- data you provided -------------------------
rows = [
    ( 1, 725, "Q:\n\n"),
    ( 2,  60, "1. Field of"),
    ( 3,  35, "---\nabstract:"),
    ( 4,  17, "/*\n * Copyright"),
    ( 5,  14, "1. Introduction {#"),
    ( 6,  14, '<?xml version="'),
    ( 7,  12, "The present invention relates"),
    ( 8,  10, "Introduction\n============\n"),
    ( 9,   9, "This invention relates to"),
    (10,   9, "If this is your"),
    (11,   9, "Introduction {#s1"),
    (12,   7, "Background {#Sec1"),
    (13,   7, "INTRODUCTION\n============"),
    (14,   6, "Background\n==========\n"),
    (15,   6, "Introduction {#Sec1"),
    (16,   6, "#!/usr/"),
    (17,   6, "Check out our new"),
    (18,   5, "The invention relates to"),
    (19,   5, "//\n//"),
    (20,   5, '{\n  "'),
]


df = pd.DataFrame(rows, columns=["rank", "count", "bigram"])

# Render literal "\n" instead of breaking lines
df["bigram_disp"] = df["bigram"].str.replace("\n", r"\n", regex=False)

# ------------------------- look & feel to match plots -------------------------
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)   # keep consistent with your figures

# Single-column-ish figure size; tweak if you place it full width
fig, ax = plt.subplots(figsize=(4.8, 10))
ax.axis("off")  # hide axes, we only want the table

# Column labels and cell text
col_labels = ["Rank", "Count", "4-Token Prefixes "]
cell_text = df[["rank", "count", "bigram_disp"]].values.tolist()

# Zebra striping: very light greys to keep it subtle in print
n = len(cell_text)
row_colors = [["#ffffff"] if i % 2 == 0 else ["#f6f6f6"] for i in range(n)]
# Expand to 3 columns (same color across each row)
cell_colours = [rc * 3 for rc in row_colors]

# Header style
header_colours = ["#e8e8e8"] * len(col_labels)

# Build the table (colWidths are proportions of the axes width)
table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    colColours=header_colours,
    cellColours=cell_colours,
    colLoc="center",
    cellLoc="center",
    colWidths=[0.17, 0.2, 1.2],
    loc="center"
)

# Typography & spacing
table.auto_set_font_size(False)
table.set_fontsize(10)        # scaled by sns.set_context above
table.scale(1.0, 1.2)         # increase row height a touch

# Left-align the Bigram column (data rows only; header stays centered)
# In matplotlib tables, (0, *) is header row when colLabels are used
for r in range(1, n + 1):
    table[r, 2].get_text().set_ha("left")

# Light borders (looks nicer in print than heavy gridlines)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#dddddd")
    cell.set_linewidth(0.6)

# Title (optional; comment out if you’ll caption in LaTeX)
#ax.set_title("Top starting bigrams in the validation set", pad=10)

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/starting_bigrams_table.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True
)
plt.show()

# ------------------------- optional: LaTeX table export -------------------------
# If you'd like a booktabs table to paste into Overleaf:
#latex = df[["rank","count","bigram_disp"]].rename(
#    columns={"rank":"Rank","count":"Count","bigram_disp":"Bigram"}
#).to_latex(index=False, escape=False, bold_rows=False, longtable=False, caption="Top starting bigrams in the validation set", label="tab:starting-bigrams", column_format="rrl", header=True)
#with open("D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/starting_bigrams_table.tex", "w", encoding="utf-8") as f:
#    f.write(latex)
