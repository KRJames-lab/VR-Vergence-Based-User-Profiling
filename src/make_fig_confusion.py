"""
Fig 3 for the IDW paper: session-to-session cluster confusion matrix (S1 -> S2).
Single-column. Counts from research_notebook.html Step 6 (N=407 session stability):
  rows = Session 1 profile, cols = Session 2 profile (nearest-cluster reassignment).
  Row sums = cluster sizes (263/118/26); diagonal = retention (81.7/52.5/26.9%).
Cell shading = row-normalized proportion; annotations = participant counts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FIG = 'my_paper/manuscript/fig_confusion.png'

LABELS = ['Normal', 'Over-\nconv.', 'Under-\nconv.']
# rows = S1, cols = S2, order [Normal, Over-converging, Under-converging]
M = np.array([[215, 43, 5],
              [55, 62, 1],
              [13, 6, 7]], dtype=float)
row_norm = M / M.sum(axis=1, keepdims=True)

plt.rcParams.update({
    'font.size': 9, 'axes.linewidth': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})

fig, ax = plt.subplots(figsize=(3.3, 2.9))
ax.imshow(row_norm, cmap='Blues', vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{int(M[i, j])}', ha='center', va='center',
                fontsize=10, color='white' if row_norm[i, j] > 0.5 else 'black')

ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(LABELS, fontsize=7)
ax.set_yticklabels(LABELS, fontsize=7)
ax.set_xlabel('Session 2 profile', fontsize=8)
ax.set_ylabel('Session 1 profile', fontsize=8)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.0)
ax.tick_params(which='minor', length=0)
fig.tight_layout()
fig.savefig(FIG, dpi=300, bbox_inches='tight')
print(f"Saved: {FIG}")

print("Row sums (cluster sizes):", M.sum(axis=1).astype(int))
print("Diagonal retention %:", (np.diag(M) / M.sum(axis=1) * 100).round(1))
print("Into Normal (col0 off-diag):", int(M[1, 0] + M[2, 0]),
      "| Out of Normal (row0 off-diag):", int(M[0, 1] + M[0, 2]))
print("Under->Normal:", int(M[2, 0]), "| Normal->Under:", int(M[0, 2]))
