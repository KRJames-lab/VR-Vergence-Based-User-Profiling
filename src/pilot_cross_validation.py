"""
Step 5: Cross-Task Validation - PUR & RAN
VRG에서 만든 군집이 PUR, RAN 태스크에서도 의미 있는 차이를 보이는지 검증합니다.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Configuration
# ============================================================
DATA_DIR = 'datasets/gazebasevr/extracted_full/data'
OUT_DIR = 'eda_results/eye_tracking/full_407'
BIO_CSV = os.path.join(OUT_DIR, 'pilot_biomarkers_with_clusters.csv')

OUTLIER_THRESHOLD = 50
VERGENCE_VELOCITY_THRESHOLD = 1.0
TARGET_CHANGE_THRESHOLD = 0.5  # degree, for RAN target change detection

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
# Preprocessing (same as biomarker extraction)
# ============================================================
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['time_s'] = df['n'] / 1000.0

    for col in ['lx', 'ly', 'rx', 'ry']:
        df.loc[df[col].abs() > OUTLIER_THRESHOLD, col] = np.nan

    df[['lx', 'ly', 'rx', 'ry']] = df[['lx', 'ly', 'rx', 'ry']].interpolate(method='linear')
    df = df.dropna(subset=['lx', 'rx']).copy()

    df['vergence'] = df['lx'] - df['rx']

    vergence_diff = df['vergence'].diff().abs()
    artifact_mask = vergence_diff > VERGENCE_VELOCITY_THRESHOLD
    df.loc[artifact_mask, 'vergence'] = np.nan
    df['vergence'] = df['vergence'].interpolate(method='linear')
    df = df.dropna(subset=['vergence']).copy()

    # Cyclopean gaze
    df['cx'] = (df['lx'] + df['rx']) / 2
    df['cy'] = (df['ly'] + df['ry']) / 2

    return df


# ============================================================
# PUR Metrics
# ============================================================
def compute_pur_metrics(df):
    """Smooth Pursuit 태스크의 시선 품질 지표를 계산합니다."""
    # xT, yT가 NaN이 아닌 구간만 사용
    valid = df.dropna(subset=['xT', 'yT']).copy()
    if len(valid) < 100:
        return None

    # 시선-타겟 오차
    error_x = valid['cx'] - valid['xT']
    error_y = valid['cy'] - valid['yT']
    error_dist = np.sqrt(error_x**2 + error_y**2)

    mae = error_dist.mean()
    mse = (error_dist**2).mean()
    rmse = np.sqrt(mse)

    # R^2 (x 방향 - PUR는 주로 수평 이동)
    ss_res = ((valid['cx'] - valid['xT'])**2).sum()
    ss_tot = ((valid['cx'] - valid['cx'].mean())**2).sum()
    r2_x = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Pursuit Gain (시선 속도 / 타겟 속도)
    dt = 1 / 250  # seconds
    gaze_vel = np.sqrt(valid['cx'].diff()**2 + valid['cy'].diff()**2) / dt
    target_vel = np.sqrt(valid['xT'].diff()**2 + valid['yT'].diff()**2) / dt

    # 타겟이 움직이는 구간만 (속도 > 1 deg/s)
    moving = target_vel > 1.0
    if moving.sum() > 10:
        gain = (gaze_vel[moving] / target_vel[moving]).median()
    else:
        gain = np.nan

    # Lag (cross-correlation으로 추정)
    cx_clean = valid['cx'].values
    xt_clean = valid['xT'].values
    # 둘 다 표준화
    cx_norm = (cx_clean - cx_clean.mean()) / (cx_clean.std() + 1e-8)
    xt_norm = (xt_clean - xt_clean.mean()) / (xt_clean.std() + 1e-8)
    max_lag_samples = 125  # 500ms max
    corr = np.correlate(cx_norm, xt_norm, mode='full')
    mid = len(corr) // 2
    # 시선이 타겟보다 뒤처지는 방향의 lag만 탐색 (양의 lag)
    search = corr[mid:mid + max_lag_samples]
    lag_samples = np.argmax(search)
    lag_ms = lag_samples * 4  # 4ms per sample

    # Vergence 안정성 (PUR 중 vergence의 std - 깊이 변화 없으므로 작을수록 좋음)
    vergence_std = valid['vergence'].std()

    return {
        'pur_mae': mae,
        'pur_rmse': rmse,
        'pur_r2_x': r2_x,
        'pur_gain': gain,
        'pur_lag_ms': lag_ms,
        'pur_vergence_std': vergence_std,
    }


# ============================================================
# RAN Metrics
# ============================================================
def compute_ran_metrics(df):
    """Random Saccade 태스크의 시선 품질 지표를 계산합니다."""
    valid = df.dropna(subset=['xT', 'yT']).copy()
    if len(valid) < 100:
        return None

    # 시선-타겟 오차
    error_x = valid['cx'] - valid['xT']
    error_y = valid['cy'] - valid['yT']
    error_dist = np.sqrt(error_x**2 + error_y**2)

    mae = error_dist.mean()
    rmse = np.sqrt((error_dist**2).mean())

    # R^2
    ss_res_x = ((valid['cx'] - valid['xT'])**2).sum()
    ss_tot_x = ((valid['cx'] - valid['cx'].mean())**2).sum()
    r2_x = 1 - ss_res_x / ss_tot_x if ss_tot_x > 0 else np.nan

    ss_res_y = ((valid['cy'] - valid['yT'])**2).sum()
    ss_tot_y = ((valid['cy'] - valid['cy'].mean())**2).sum()
    r2_y = 1 - ss_res_y / ss_tot_y if ss_tot_y > 0 else np.nan

    # 타겟 전환 시점 찾기
    target_diff = np.sqrt(valid['xT'].diff()**2 + valid['yT'].diff()**2)
    transitions = valid.index[target_diff > TARGET_CHANGE_THRESHOLD].tolist()

    # Saccade 후 vergence 안정화 시간
    stabilization_times = []
    vergence_stds_after = []

    for i, t_idx in enumerate(transitions):
        next_idx = transitions[i + 1] if i + 1 < len(transitions) else valid.index[-1]
        segment = valid.loc[t_idx:next_idx]

        if len(segment) < 50:  # 최소 200ms
            continue

        seg_vergence = segment['vergence'].values
        seg_time_ms = (segment['time_s'].values - segment['time_s'].values[0]) * 1000

        # 타겟 도달 판정: 시선-타겟 오차 < 2도
        seg_error = np.sqrt(
            (segment['cx'].values - segment['xT'].values)**2 +
            (segment['cy'].values - segment['yT'].values)**2
        )

        # 도달 시점 찾기
        arrival_idx = None
        for j in range(len(seg_error)):
            if seg_error[j] < 2.0:
                arrival_idx = j
                break

        if arrival_idx is None or arrival_idx >= len(seg_vergence) - 25:
            continue

        # 도달 후 vergence 안정화 시간 (rolling std < 0.5도가 100ms 유지)
        after_arrival = seg_vergence[arrival_idx:]
        after_time = seg_time_ms[arrival_idx:]
        rolling_std = pd.Series(after_arrival).rolling(window=25, min_periods=10).std().values

        stab_time = np.nan
        for j in range(len(rolling_std)):
            if not np.isnan(rolling_std[j]) and rolling_std[j] < 0.5:
                stab_time = after_time[j] - after_time[0]
                break

        if not np.isnan(stab_time):
            stabilization_times.append(stab_time)

        # 도달 후 200ms 구간의 vergence std
        end_idx = min(arrival_idx + 50, len(seg_vergence))
        if end_idx - arrival_idx > 10:
            vergence_stds_after.append(np.std(seg_vergence[arrival_idx:end_idx]))

    # Vergence 전체 안정성
    vergence_std = valid['vergence'].std()

    return {
        'ran_mae': mae,
        'ran_rmse': rmse,
        'ran_r2_x': r2_x,
        'ran_r2_y': r2_y,
        'ran_vergence_std': vergence_std,
        'ran_stab_time_ms': np.mean(stabilization_times) if stabilization_times else np.nan,
        'ran_stab_vergence_std': np.mean(vergence_stds_after) if vergence_stds_after else np.nan,
        'ran_n_transitions': len(transitions),
    }


# ============================================================
# Main
# ============================================================
print("=" * 70)
print("  Step 5: Cross-Task Validation (PUR & RAN)")
print("=" * 70)

# Load cluster labels
df_bio = pd.read_csv(BIO_CSV)
print(f"Loaded {len(df_bio)} participants with cluster labels")

# K=3 clusters
cluster_col = 'cluster_k3'
n_clusters = 3
print(f"Using {cluster_col}: {df_bio[cluster_col].value_counts().to_dict()}")

# ============================================================
# Process PUR files
# ============================================================
print("\n--- Processing PUR (Smooth Pursuit) ---")
pur_results = []
for _, row in df_bio.iterrows():
    pid = str(int(row['participant']))
    session = row['session']

    # Find matching PUR file
    pattern = os.path.join(DATA_DIR, f'S_{pid}_{session}_2_PUR.csv')
    matches = glob.glob(pattern)

    if not matches:
        continue

    df_pur = load_and_preprocess(matches[0])
    metrics = compute_pur_metrics(df_pur)

    if metrics:
        metrics['participant'] = int(pid)
        metrics['cluster'] = row[cluster_col]
        pur_results.append(metrics)

df_pur_results = pd.DataFrame(pur_results)
print(f"  PUR metrics computed for {len(df_pur_results)} participants")

# ============================================================
# Process RAN files
# ============================================================
print("\n--- Processing RAN (Random Saccade) ---")
ran_results = []
for _, row in df_bio.iterrows():
    pid = str(int(row['participant']))
    session = row['session']

    pattern = os.path.join(DATA_DIR, f'S_{pid}_{session}_5_RAN.csv')
    matches = glob.glob(pattern)

    if not matches:
        continue

    df_ran = load_and_preprocess(matches[0])
    metrics = compute_ran_metrics(df_ran)

    if metrics:
        metrics['participant'] = int(pid)
        metrics['cluster'] = row[cluster_col]
        ran_results.append(metrics)

df_ran_results = pd.DataFrame(ran_results)
print(f"  RAN metrics computed for {len(df_ran_results)} participants")

# ============================================================
# Statistical Tests
# ============================================================
print("\n" + "=" * 70)
print("  Statistical Tests: Cluster Differences in PUR & RAN")
print("=" * 70)


def test_cluster_difference(df, metric_col, metric_name, n_clusters):
    """군집 간 차이를 검정합니다."""
    groups = [df.loc[df['cluster'] == c, metric_col].dropna() for c in range(n_clusters)]
    groups = [g for g in groups if len(g) >= 3]

    if len(groups) < 2:
        return None

    # 정규성 검정
    normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)

    if len(groups) == 2:
        if normal:
            stat, p = stats.ttest_ind(groups[0], groups[1])
            test_name = "t-test"
        else:
            stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            test_name = "Mann-Whitney U"
    else:
        if normal:
            stat, p = stats.f_oneway(*groups)
            test_name = "ANOVA"
        else:
            stat, p = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    means = [f"C{i}={g.mean():.2f}" for i, g in enumerate(groups)]
    print(f"  {metric_name:<30} {test_name:<16} stat={stat:8.3f}  p={p:.4f} {sig:<4}  {' | '.join(means)}")

    return {'metric': metric_name, 'test': test_name, 'stat': stat, 'p': p, 'sig': sig}


# PUR tests
print("\n--- PUR (Smooth Pursuit) ---")
pur_metrics = ['pur_mae', 'pur_rmse', 'pur_r2_x', 'pur_gain', 'pur_lag_ms', 'pur_vergence_std']
pur_labels = ['MAE (deg)', 'RMSE (deg)', 'R2 (horizontal)', 'Pursuit Gain', 'Lag (ms)', 'Vergence Std (deg)']
pur_test_results = []
for col, label in zip(pur_metrics, pur_labels):
    result = test_cluster_difference(df_pur_results, col, f"PUR {label}", n_clusters)
    if result:
        pur_test_results.append(result)

# RAN tests
print("\n--- RAN (Random Saccade) ---")
ran_metrics = ['ran_mae', 'ran_rmse', 'ran_r2_x', 'ran_r2_y', 'ran_vergence_std', 'ran_stab_time_ms', 'ran_stab_vergence_std']
ran_labels = ['MAE (deg)', 'RMSE (deg)', 'R2 (horizontal)', 'R2 (vertical)', 'Vergence Std (deg)', 'Stabilization Time (ms)', 'Post-saccade Verg Std (deg)']
ran_test_results = []
for col, label in zip(ran_metrics, ran_labels):
    result = test_cluster_difference(df_ran_results, col, f"RAN {label}", n_clusters)
    if result:
        ran_test_results.append(result)

# ============================================================
# Visualization
# ============================================================
print("\n--- Generating Visualizations ---")

# PUR: Box plots by cluster
pur_plot_metrics = ['pur_mae', 'pur_r2_x', 'pur_gain', 'pur_lag_ms', 'pur_vergence_std']
pur_plot_labels = ['MAE (deg)', 'R2 (horizontal)', 'Pursuit Gain', 'Lag (ms)', 'Vergence Std (deg)']

fig, axes = plt.subplots(1, len(pur_plot_metrics), figsize=(20, 5))
fig.suptitle(f'PUR (Smooth Pursuit): Cluster Comparison (K=3)', fontsize=14, fontweight='bold', color='white')

for idx, (col, label) in enumerate(zip(pur_plot_metrics, pur_plot_labels)):
    ax = axes[idx]
    box_data = [df_pur_results.loc[df_pur_results['cluster'] == c, col].dropna().values for c in range(n_clusters)]
    box_data = [d for d in box_data if len(d) > 0]

    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#8b949e'),
                    capprops=dict(color='#8b949e'),
                    flierprops=dict(marker='o', markerfacecolor='#f85149', markersize=4))
    for c, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.7)
    ax.set_xticklabels([f'C{c}' for c in range(len(box_data))])
    ax.set_title(label, fontsize=10, fontweight='bold')

    # p-value annotation
    test_result = next((r for r in pur_test_results if label in r['metric']), None)
    if test_result:
        ax.text(0.95, 0.95, f"p={test_result['p']:.3f} {test_result['sig']}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                color='#3fb950' if test_result['p'] < 0.05 else '#f85149')

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(OUT_DIR, 'pilot_crossval_pur.png'), bbox_inches='tight')
plt.close()
print("Saved: pilot_crossval_pur.png")

# RAN: Box plots by cluster
ran_plot_metrics = ['ran_mae', 'ran_r2_x', 'ran_vergence_std', 'ran_stab_time_ms', 'ran_stab_vergence_std']
ran_plot_labels = ['MAE (deg)', 'R2 (horizontal)', 'Vergence Std (deg)', 'Stabilization (ms)', 'Post-sacc Verg Std (deg)']

fig, axes = plt.subplots(1, len(ran_plot_metrics), figsize=(20, 5))
fig.suptitle(f'RAN (Random Saccade): Cluster Comparison (K=3)', fontsize=14, fontweight='bold', color='white')

for idx, (col, label) in enumerate(zip(ran_plot_metrics, ran_plot_labels)):
    ax = axes[idx]
    box_data = [df_ran_results.loc[df_ran_results['cluster'] == c, col].dropna().values for c in range(n_clusters)]
    box_data = [d for d in box_data if len(d) > 0]

    if all(len(d) == 0 for d in box_data):
        ax.set_title(f'{label}\n(no data)', fontsize=10)
        continue

    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#8b949e'),
                    capprops=dict(color='#8b949e'),
                    flierprops=dict(marker='o', markerfacecolor='#f85149', markersize=4))
    for c, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.7)
    ax.set_xticklabels([f'C{c}' for c in range(len(box_data))])
    ax.set_title(label, fontsize=10, fontweight='bold')

    test_result = next((r for r in ran_test_results if label in r['metric']), None)
    if test_result:
        ax.text(0.95, 0.95, f"p={test_result['p']:.3f} {test_result['sig']}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                color='#3fb950' if test_result['p'] < 0.05 else '#f85149')

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(OUT_DIR, 'pilot_crossval_ran.png'), bbox_inches='tight')
plt.close()
print("Saved: pilot_crossval_ran.png")

# ============================================================
# Save results
# ============================================================
df_pur_results.to_csv(os.path.join(OUT_DIR, 'pilot_pur_metrics.csv'), index=False)
df_ran_results.to_csv(os.path.join(OUT_DIR, 'pilot_ran_metrics.csv'), index=False)

all_tests = pur_test_results + ran_test_results
df_tests = pd.DataFrame(all_tests)
df_tests.to_csv(os.path.join(OUT_DIR, 'pilot_crossval_tests.csv'), index=False)
print(f"\nSaved test results: pilot_crossval_tests.csv")

# Summary
print("\n" + "=" * 70)
print("  Summary")
print("=" * 70)
sig_count = sum(1 for t in all_tests if t['p'] < 0.05)
total_count = len(all_tests)
print(f"  Significant results (p<0.05): {sig_count}/{total_count}")
for t in all_tests:
    if t['p'] < 0.05:
        print(f"    {t['sig']} {t['metric']}: p={t['p']:.4f}")

print("\nCross-validation complete!")
