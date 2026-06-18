"""
VRG Vergence Biomarker Clustering - Pilot (42 participants)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
OUT_DIR = 'eda_results/eye_tracking/full_407'
BIO_CSV = os.path.join(OUT_DIR, 'pilot_biomarkers.csv')

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.3,
    'axes.grid': True,
    'savefig.dpi': 150,
    'savefig.facecolor': '#0d1117',
})

COLORS = ['#e94560', '#00b4d8', '#2a9d8f', '#e9c46a', '#533483']

# ============================================================
# Load Data
# ============================================================
print("=" * 70)
print("  VRG Vergence Biomarker Clustering - Pilot")
print("=" * 70)

df = pd.read_csv(BIO_CSV)
print(f"\nLoaded {len(df)} participants")

# Features for clustering: Conv/Div 분리 + Jitter (Peak Velocity 제외 - 계산 문제로 별도 보고)
feature_cols = [
    'conv_latency_ms',
    'conv_settling_ms',
    'conv_overshoot_pct',
    'conv_ss_error',
    'conv_speed',
    'div_latency_ms',
    'div_settling_ms',
    'div_overshoot_pct',
    'div_ss_error',
    'div_speed',
    'jitter_mean',
]
feature_labels = [
    'Conv Lat', 'Conv Set', 'Conv Over', 'Conv SSE', 'Conv Spd',
    'Div Lat', 'Div Set', 'Div Over', 'Div SSE', 'Div Spd',
    'Jitter',
]

# NaN handling: fill with max value per column (worst case)
for col in feature_cols:
    max_val = df[col].max()
    df[col] = df[col].fillna(max_val)

X = df[feature_cols].values
print(f"Feature matrix: {X.shape}")
print(f"Features: {feature_cols}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 1. Find Optimal K
# ============================================================
print("\n--- Finding Optimal K ---")

k_range = range(2, 8)
results = {
    'k': [],
    'kmeans_silhouette': [], 'kmeans_calinski': [], 'kmeans_inertia': [],
    'gmm_silhouette': [], 'gmm_bic': [],
    'agg_silhouette': [],
}

for k in k_range:
    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km_labels = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    km_cal = calinski_harabasz_score(X_scaled, km_labels)

    # GMM
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5, covariance_type='full')
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)
    gmm_bic = gmm.bic(X_scaled)

    # Agglomerative (Ward)
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)

    results['k'].append(k)
    results['kmeans_silhouette'].append(km_sil)
    results['kmeans_calinski'].append(km_cal)
    results['kmeans_inertia'].append(km.inertia_)
    results['gmm_silhouette'].append(gmm_sil)
    results['gmm_bic'].append(gmm_bic)
    results['agg_silhouette'].append(agg_sil)

    print(f"  k={k}: KMeans sil={km_sil:.3f} cal={km_cal:.1f} | "
          f"GMM sil={gmm_sil:.3f} BIC={gmm_bic:.1f} | "
          f"Agg sil={agg_sil:.3f}")

df_results = pd.DataFrame(results)

# ============================================================
# 2. Visualization: Optimal K Selection
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Clustering: Optimal K Selection (N={len(df)})',
             fontsize=16, fontweight='bold', color='white')

# Silhouette scores
ax = axes[0, 0]
ax.plot(df_results['k'], df_results['kmeans_silhouette'], 'o-', color='#e94560', label='K-Means', linewidth=2)
ax.plot(df_results['k'], df_results['gmm_silhouette'], 's-', color='#00b4d8', label='GMM', linewidth=2)
ax.plot(df_results['k'], df_results['agg_silhouette'], '^-', color='#2a9d8f', label='Agglomerative', linewidth=2)
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score (higher = better)', fontweight='bold')
ax.legend()

# Calinski-Harabasz
ax = axes[0, 1]
ax.plot(df_results['k'], df_results['kmeans_calinski'], 'o-', color='#e94560', linewidth=2)
ax.set_xlabel('K')
ax.set_ylabel('Calinski-Harabasz Index')
ax.set_title('Calinski-Harabasz (K-Means, higher = better)', fontweight='bold')

# Elbow (Inertia)
ax = axes[0, 2]
ax.plot(df_results['k'], df_results['kmeans_inertia'], 'o-', color='#e9c46a', linewidth=2)
ax.set_xlabel('K')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method (K-Means, lower = better)', fontweight='bold')

# GMM BIC
ax = axes[1, 0]
ax.plot(df_results['k'], df_results['gmm_bic'], 's-', color='#00b4d8', linewidth=2)
ax.set_xlabel('K')
ax.set_ylabel('BIC')
ax.set_title('BIC (GMM, lower = better)', fontweight='bold')

# Best K summary table
ax = axes[1, 1]
ax.axis('off')
best_km_k = df_results.loc[df_results['kmeans_silhouette'].idxmax(), 'k']
best_gmm_k = df_results.loc[df_results['gmm_bic'].idxmin(), 'k']
best_agg_k = df_results.loc[df_results['agg_silhouette'].idxmax(), 'k']

text = f"""Best K by Metric:

  K-Means Silhouette:  k = {best_km_k}  (sil = {df_results['kmeans_silhouette'].max():.3f})
  GMM BIC:             k = {best_gmm_k}  (BIC = {df_results['gmm_bic'].min():.1f})
  Agglom. Silhouette:  k = {best_agg_k}  (sil = {df_results['agg_silhouette'].max():.3f})
  Calinski-Harabasz:   k = {df_results.loc[df_results['kmeans_calinski'].idxmax(), 'k']}  (CH = {df_results['kmeans_calinski'].max():.1f})
"""
ax.text(0.1, 0.5, text, fontsize=12, fontfamily='monospace',
        color='white', transform=ax.transAxes, va='center')
ax.set_title('Summary', fontweight='bold')

# Empty
axes[1, 2].axis('off')

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, 'pilot_optimal_k.png'), bbox_inches='tight')
plt.close()
print(f"\nSaved: pilot_optimal_k.png")

# ============================================================
# 3. Run clustering with multiple K values (2, 3, 4)
# ============================================================
print("\n--- Running Clustering for K=2,3,4 ---")

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%} = {sum(pca.explained_variance_ratio_):.2%}")

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
fig.suptitle('Clustering Results: K=2, 3, 4 (PCA + Biomarker Profiles)',
             fontsize=16, fontweight='bold', color='white')

for row, k in enumerate([2, 3, 4]):
    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    df[f'cluster_k{k}'] = labels

    # PCA scatter
    ax = axes[row, 0]
    for c in range(k):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[c],
                   label=f'Cluster {c} (n={mask.sum()})', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'K={k} | Silhouette={sil:.3f}', fontweight='bold')
    ax.legend(fontsize=8)

    # Cluster profiles (radar-style bar chart)
    ax = axes[row, 1]
    cluster_means = []
    for c in range(k):
        mask = labels == c
        means = X_scaled[mask].mean(axis=0)
        cluster_means.append(means)

    x_pos = np.arange(len(feature_labels))
    width = 0.8 / k
    for c in range(k):
        offset = (c - k / 2 + 0.5) * width
        ax.bar(x_pos + offset, cluster_means[c], width, color=COLORS[c],
               label=f'Cluster {c}', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Standardized Value')
    ax.set_title(f'K={k} Cluster Profiles', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(0, color='#8b949e', linewidth=0.5, linestyle='--')

    # Cluster stats table (raw values)
    ax = axes[row, 2]
    ax.axis('off')
    table_text = f"{'Cluster':<10}"
    for fl in ['Lat(ms)', 'Set(ms)', 'Over(%)', 'SSE(d)', 'Jit(d)', 'MAE(d)', 'N']:
        table_text += f"{fl:>10}"
    table_text += "\n" + "-" * 85 + "\n"

    for c in range(k):
        mask = labels == c
        sub = df[mask]
        table_text += f"{'  C' + str(c):<10}"
        table_text += f"{sub['latency_ms_mean'].mean():>10.0f}"
        table_text += f"{sub['settling_ms_mean'].mean():>10.0f}"
        table_text += f"{sub['overshoot_pct_mean'].mean():>10.1f}"
        table_text += f"{sub['ss_error_mean'].mean():>10.2f}"
        table_text += f"{sub['jitter_mean'].mean():>10.2f}"
        table_text += f"{sub['vergence_mae'].mean():>10.2f}"
        table_text += f"{mask.sum():>10d}"
        table_text += "\n"

    ax.text(0.05, 0.5, table_text, fontsize=10, fontfamily='monospace',
            color='white', transform=ax.transAxes, va='center')
    ax.set_title(f'K={k} Cluster Statistics (Raw Values)', fontweight='bold')

    # Box plots per cluster (MAE)
    ax = axes[row, 3]
    box_data = [df.loc[labels == c, 'vergence_mae'].values for c in range(k)]
    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#8b949e'),
                    capprops=dict(color='#8b949e'),
                    flierprops=dict(marker='o', markerfacecolor='#f85149', markersize=5))
    for c, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.7)
    ax.set_xticklabels([f'C{c}' for c in range(k)])
    ax.set_ylabel('MAE (deg)')
    ax.set_title(f'K={k} MAE Distribution by Cluster', fontweight='bold')

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, 'pilot_clustering_results.png'), bbox_inches='tight')
plt.close()
print("Saved: pilot_clustering_results.png")

# ============================================================
# 4. Save cluster labels
# ============================================================
cluster_csv = os.path.join(OUT_DIR, 'pilot_biomarkers_with_clusters.csv')
df.to_csv(cluster_csv, index=False)
print(f"Saved: {cluster_csv}")

# ============================================================
# 5. Print detailed summary
# ============================================================
print("\n" + "=" * 70)
print("  Detailed Cluster Summary")
print("=" * 70)

for k in [2, 3, 4]:
    col = f'cluster_k{k}'
    print(f"\n--- K = {k} ---")
    for c in range(k):
        mask = df[col] == c
        sub = df[mask]
        n = mask.sum()
        print(f"\n  Cluster {c} (n={n}, {n/len(df)*100:.0f}%):")
        print(f"    Latency:    {sub['latency_ms_mean'].mean():.0f} +/- {sub['latency_ms_mean'].std():.0f} ms")
        print(f"    Settling:   {sub['settling_ms_mean'].mean():.0f} +/- {sub['settling_ms_mean'].std():.0f} ms")
        print(f"    Overshoot:  {sub['overshoot_pct_mean'].mean():.1f} +/- {sub['overshoot_pct_mean'].std():.1f} %")
        print(f"    SS Error:   {sub['ss_error_mean'].mean():.2f} +/- {sub['ss_error_mean'].std():.2f} deg")
        print(f"    Jitter:     {sub['jitter_mean'].mean():.2f} +/- {sub['jitter_mean'].std():.2f} deg")
        print(f"    Speed:      {sub['speed_mean'].mean():.1f} +/- {sub['speed_mean'].std():.1f} deg/s")
        print(f"    MAE:        {sub['vergence_mae'].mean():.2f} +/- {sub['vergence_mae'].std():.2f} deg")
        print(f"    Participants: {sorted(sub['participant'].tolist())}")

print("\nPilot clustering complete!")
