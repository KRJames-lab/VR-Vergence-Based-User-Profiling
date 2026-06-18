"""
Two figures for the IDW paper, built from pilot_biomarkers_with_clusters.csv:

  fig_clusters.png  (Results) : PCA scatter of the K=3 clusters + MAE boxplot
  fig_conv_div.png  (Sec 2.2) : convergence vs divergence boxplots
                                (latency / settling / overshoot)

Also prints the per-cluster biomarker means for Table 2.
Cluster labels (cluster_k3): 0 = over-converging (118), 1 = normal (263),
2 = under-converging (26).  (Renamed 2026-06 from reach-failure/unstable.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = 'eda_results/eye_tracking/full_407'
CSV = os.path.join(OUT_DIR, 'pilot_biomarkers_with_clusters.csv')
CSV_RAW = os.path.join(OUT_DIR, 'pilot_biomarkers.csv')  # pre-fillna original, for conv/div
FIG_CLUSTERS = 'my_paper/manuscript/fig_clusters.png'
FIG_CONVDIV = 'my_paper/manuscript/fig_conv_div.png'

feature_cols = [
    'conv_latency_ms', 'conv_settling_ms', 'conv_overshoot_pct', 'conv_ss_error', 'conv_speed',
    'div_latency_ms', 'div_settling_ms', 'div_overshoot_pct', 'div_ss_error', 'div_speed',
    'jitter_mean',
]

LABELS = {1: 'Normal', 0: 'Over-converging', 2: 'Under-converging'}
COLORS = {1: '#1f77b4', 0: '#ff7f0e', 2: '#d62728'}
MARKERS = {1: 'o', 0: 's', 2: '^'}
ORDER = [1, 0, 2]

plt.rcParams.update({
    'font.size': 9, 'axes.linewidth': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})

df = pd.read_csv(CSV)
for c in feature_cols:
    df[c] = df[c].fillna(df[c].max())
X = StandardScaler().fit_transform(df[feature_cols].values)
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
ev = pca.explained_variance_ratio_ * 100
labels = df['cluster_k3'].values

# ---------------------------------------------------------------- Figure: clusters
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

ax = axes[0]
for c in ORDER:
    m = labels == c
    ax.scatter(Xp[m, 0], Xp[m, 1], c=COLORS[c], marker=MARKERS[c], s=16,
               alpha=0.6, edgecolors='none', label=LABELS[c])
ax.set_xlabel(f'PC1 ({ev[0]:.0f}%)')
ax.set_ylabel(f'PC2 ({ev[1]:.0f}%)')
ax.set_title('Cluster separation (PCA)', fontsize=9)
ax.grid(True, color='0.92', linewidth=0.5)

ax = axes[1]
data = [df.loc[labels == c, 'vergence_mae'].values for c in ORDER]
bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=1.2),
                flierprops=dict(marker='o', markersize=4.0, alpha=0.55,
                                markerfacecolor='0.3', markeredgecolor='none'))
for patch, c in zip(bp['boxes'], ORDER):
    patch.set_facecolor(COLORS[c])
    patch.set_alpha(0.6)
ax.set_xticklabels([LABELS[c] for c in ORDER], rotation=12, fontsize=7)
ax.set_ylabel('Tracking error, MAE (deg)')
ax.set_title('Error by profile', fontsize=9)
ax.grid(True, axis='y', color='0.92', linewidth=0.5)

fig.tight_layout()
fig.savefig(FIG_CLUSTERS, dpi=300, bbox_inches='tight')
print(f"Saved: {FIG_CLUSTERS}")

# ---------------------------------------------------------------- Figure: conv/div (ORIGINAL data, pre-fillna)
df_raw = pd.read_csv(CSV_RAW)
fig2, axes2 = plt.subplots(1, 3, figsize=(3.4, 2.6))
metrics = [('conv_latency_ms', 'div_latency_ms', 'Latency (ms)'),
           ('conv_settling_ms', 'div_settling_ms', 'Settling (ms)'),
           ('conv_overshoot_pct', 'div_overshoot_pct', 'Overshoot (%)')]
for ax, (cc, dc, title) in zip(axes2, metrics):
    d = [df_raw[cc].dropna().values, df_raw[dc].dropna().values]
    bp = ax.boxplot(d, patch_artist=True, widths=0.6, showfliers=True,
                    medianprops=dict(color='black', linewidth=1.0),
                    flierprops=dict(marker='o', markersize=4.0, alpha=0.55,
                                    markerfacecolor='0.3', markeredgecolor='none'))
    bp['boxes'][0].set_facecolor('#4c72b0'); bp['boxes'][0].set_alpha(0.65)
    bp['boxes'][1].set_facecolor('#c44e52'); bp['boxes'][1].set_alpha(0.65)
    ax.set_xticklabels(['Conv', 'Div'], fontsize=6.5)
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(True, axis='y', color='0.92', linewidth=0.5)
fig2.tight_layout()
fig2.savefig(FIG_CONVDIV, dpi=300, bbox_inches='tight')
print(f"Saved: {FIG_CONVDIV}")

# ---------------------------------------------------------------- Table 2 values
print(f"\nPCA explained variance: PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%  (sum {ev[0]+ev[1]:.1f}%)")
print(f"\n{'Profile':<16}{'n':>5}{'Latency':>9}{'Settling':>9}{'Overshoot':>10}{'SSerr':>7}{'Jitter':>7}{'MAE':>6}")
for c in ORDER:
    s = df[labels == c]
    print(f"{LABELS[c]:<16}{len(s):5d}{s['latency_ms_mean'].mean():9.0f}"
          f"{s['settling_ms_mean'].mean():9.0f}{s['overshoot_pct_mean'].mean():10.0f}"
          f"{s['ss_error_mean'].mean():7.2f}{s['jitter_mean'].mean():7.2f}{s['vergence_mae'].mean():6.2f}")
