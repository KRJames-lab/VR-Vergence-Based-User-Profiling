"""
Step 6: Session Stability - S1 군집이 S2에서도 재현되는가?
S1에서 추출한 바이오마커로 군집화 → S2에서 동일 바이오마커 추출 → 같은 군집에 배정되는지 검증
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = 'datasets/gazebasevr/extracted_full/data'
OUT_DIR = 'eda_results/eye_tracking/full_407'
S1_BIO_CSV = os.path.join(OUT_DIR, 'pilot_biomarkers_with_clusters.csv')

OUTLIER_THRESHOLD = 50
VERGENCE_VELOCITY_THRESHOLD = 1.0
SETTLING_BAND_DEG = 1.0
SETTLING_HOLD_MS = 200
STEADY_STATE_START_MS = 1500
SMOOTHING_WINDOW = 13
SAMPLING_RATE = 250
DT_MS = 1000 / SAMPLING_RATE
DT_S = 1.0 / SAMPLING_RATE
BUTTER_ORDER = 2
BUTTER_CUTOFF_HZ = 40
VELOCITY_ONSET_THRESHOLD = 1.5
LATENCY_SIGMA = 2

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
# Reuse preprocessing and biomarker functions from pilot_biomarker_extraction.py
# ============================================================
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['time_s'] = df['n'] / 1000.0
    for col in ['lx', 'ly', 'rx', 'ry']:
        df.loc[df[col].abs() > OUTLIER_THRESHOLD, col] = np.nan
    df[['lx', 'ly', 'rx', 'ry']] = df[['lx', 'ly', 'rx', 'ry']].interpolate(method='linear')
    df = df.dropna(subset=['lx', 'rx', 'zT']).copy()
    df['vergence'] = df['lx'] - df['rx']
    vergence_diff = df['vergence'].diff().abs()
    artifact_mask = vergence_diff > VERGENCE_VELOCITY_THRESHOLD
    df.loc[artifact_mask, 'vergence'] = np.nan
    df['vergence'] = df['vergence'].interpolate(method='linear')
    df = df.dropna(subset=['vergence']).copy()

    # Butterworth 2차 40Hz LP 필터
    b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ, btype='low', fs=SAMPLING_RATE)
    if len(df) > 3 * max(len(b), len(a)):
        df['vergence_filt'] = filtfilt(b, a, df['vergence'].values)
    else:
        df['vergence_filt'] = df['vergence'].values

    # TPCD 속도
    verg = df['vergence_filt'].values
    velocity = np.zeros(len(verg))
    velocity[1:-1] = (verg[2:] - verg[:-2]) / (2 * DT_S)
    velocity[0] = (verg[1] - verg[0]) / DT_S
    velocity[-1] = (verg[-1] - verg[-2]) / DT_S
    df['vergence_velocity'] = velocity

    ipd = (df['crx'] - df['clx']).median()
    df['ideal_vergence'] = 2 * np.degrees(np.arctan((ipd / 2) / df['zT']))
    df['error'] = df['vergence_filt'] - df['ideal_vergence']
    return df, ipd


def find_transitions(df):
    zT_diff = df['zT'].diff().abs()
    return df.index[zT_diff > 0.01].tolist()


def extract_step_biomarkers(df, transition_idx, next_transition_idx=None):
    if next_transition_idx is None:
        segment = df.loc[transition_idx:].copy()
    else:
        segment = df.loc[transition_idx:next_transition_idx - 1].copy()
    if len(segment) < 20:
        return None
    t0 = segment['time_s'].iloc[0]
    segment['t_rel'] = segment['time_s'] - t0
    segment['t_rel_ms'] = segment['t_rel'] * 1000
    ideal = segment['ideal_vergence'].iloc[-1]
    v_before = segment['vergence_filt'].iloc[0]
    delta = ideal - v_before
    if abs(delta) < 0.5:
        return None
    direction = 'converge' if delta > 0 else 'diverge'
    vergence = segment['vergence_filt'].values
    velocity = segment['vergence_velocity'].values
    t_ms = segment['t_rel_ms'].values

    # Latency (raw vergence에서 2σ 변화 시점)
    LATENCY_SIGMA = 2
    raw_vergence = segment['vergence'].values
    pre_window = raw_vergence[:max(1, int(50 / DT_MS))]
    pre_mean = np.mean(pre_window)
    pre_std = np.std(pre_window) if len(pre_window) > 1 else 0.5
    threshold = LATENCY_SIGMA * max(pre_std, 0.3)
    latency_ms = np.nan
    for i in range(len(raw_vergence)):
        if abs(raw_vergence[i] - pre_mean) > threshold:
            latency_ms = t_ms[i]
            break

    # Settling
    settling_ms = np.nan
    hold_samples = int(SETTLING_HOLD_MS / DT_MS)
    for i in range(len(vergence) - hold_samples):
        window = vergence[i:i + hold_samples]
        if np.all(np.abs(window - ideal) <= SETTLING_BAND_DEG):
            settling_ms = t_ms[i]
            break

    # Overshoot
    if direction == 'converge':
        peak = np.max(vergence)
        overshoot_abs = peak - ideal
    else:
        peak = np.min(vergence)
        overshoot_abs = ideal - peak
    overshoot_pct = max(0, (overshoot_abs / abs(delta)) * 100) if abs(delta) > 0 else 0

    # SS Error & Jitter
    ss_start_idx = np.searchsorted(t_ms, STEADY_STATE_START_MS)
    if ss_start_idx < len(vergence) - 5:
        ss_vergence = vergence[ss_start_idx:]
        ss_error = np.mean(np.abs(ss_vergence - ideal))
        ss_std = np.std(ss_vergence)
    else:
        ss_error = np.nan
        ss_std = np.nan

    # Speed
    target_90 = v_before + 0.9 * delta
    speed = np.nan
    for i in range(len(vergence)):
        if direction == 'converge' and vergence[i] >= target_90:
            if t_ms[i] > 0:
                speed = abs(0.9 * delta) / (t_ms[i] / 1000)
            break
        elif direction == 'diverge' and vergence[i] <= target_90:
            if t_ms[i] > 0:
                speed = abs(0.9 * delta) / (t_ms[i] / 1000)
            break

    # Peak Velocity (TPCD velocity에서 onset~offset 구간의 max)
    onset_idx = None
    offset_idx = None
    for i in range(len(velocity)):
        if abs(velocity[i]) > VELOCITY_ONSET_THRESHOLD:
            if onset_idx is None:
                onset_idx = i
            offset_idx = i
    if onset_idx is not None and offset_idx is not None and offset_idx > onset_idx:
        peak_velocity = np.max(np.abs(velocity[onset_idx:offset_idx + 1]))
    else:
        peak_velocity = np.max(np.abs(velocity)) if len(velocity) > 0 else np.nan

    return {
        'direction': direction, 'delta': delta,
        'latency_ms': latency_ms, 'settling_ms': settling_ms,
        'overshoot_pct': overshoot_pct, 'ss_error': ss_error,
        'jitter': ss_std, 'speed_deg_s': speed,
        'peak_velocity': peak_velocity,
    }


def extract_participant_biomarkers(fpath):
    """한 VRG 파일에서 참가자 바이오마커를 추출합니다."""
    df, ipd = load_and_preprocess(fpath)
    transitions = find_transitions(df)
    step_results = []
    for i, t_idx in enumerate(transitions):
        next_idx = transitions[i + 1] if i + 1 < len(transitions) else None
        bio = extract_step_biomarkers(df, t_idx, next_idx)
        if bio is not None:
            step_results.append(bio)
    if len(step_results) == 0:
        return None
    df_steps = pd.DataFrame(step_results)
    conv = df_steps[df_steps['direction'] == 'converge']
    div = df_steps[df_steps['direction'] == 'diverge']
    return {
        'conv_latency_ms': conv['latency_ms'].mean() if len(conv) > 0 else np.nan,
        'conv_settling_ms': conv['settling_ms'].mean() if len(conv) > 0 else np.nan,
        'conv_overshoot_pct': conv['overshoot_pct'].mean() if len(conv) > 0 else np.nan,
        'conv_ss_error': conv['ss_error'].mean() if len(conv) > 0 else np.nan,
        'conv_speed': conv['speed_deg_s'].mean() if len(conv) > 0 else np.nan,
        'conv_peak_velocity': conv['peak_velocity'].mean() if len(conv) > 0 else np.nan,
        'div_latency_ms': div['latency_ms'].mean() if len(div) > 0 else np.nan,
        'div_settling_ms': div['settling_ms'].mean() if len(div) > 0 else np.nan,
        'div_overshoot_pct': div['overshoot_pct'].mean() if len(div) > 0 else np.nan,
        'div_ss_error': div['ss_error'].mean() if len(div) > 0 else np.nan,
        'div_speed': div['speed_deg_s'].mean() if len(div) > 0 else np.nan,
        'div_peak_velocity': div['peak_velocity'].mean() if len(div) > 0 else np.nan,
        'jitter_mean': df_steps['jitter'].mean(),
        'vergence_mae': np.abs(df['error']).mean(),
    }


# ============================================================
# Main
# ============================================================
print("=" * 70)
print("  Step 6: Session Stability (S1 vs S2)")
print("=" * 70)

# 1. Load S1 results with cluster labels
df_s1 = pd.read_csv(S1_BIO_CSV)
print(f"S1 participants: {len(df_s1)}")
print(f"S1 K=3 distribution: {df_s1['cluster_k3'].value_counts().to_dict()}")

# 2. Get S1 scaler and K-Means model (refit)
feature_cols = [
    'conv_latency_ms', 'conv_settling_ms', 'conv_overshoot_pct', 'conv_ss_error', 'conv_speed',
    'div_latency_ms', 'div_settling_ms', 'div_overshoot_pct', 'div_ss_error', 'div_speed',
    'jitter_mean',
]

# Fill NaN with max (same as clustering)
for col in feature_cols:
    max_val = df_s1[col].max()
    df_s1[col] = df_s1[col].fillna(max_val)

X_s1 = df_s1[feature_cols].values
scaler = StandardScaler()
X_s1_scaled = scaler.fit_transform(X_s1)

km = KMeans(n_clusters=3, random_state=42, n_init=20)
s1_labels = km.fit_predict(X_s1_scaled)

# 3. Extract S2 biomarkers
print("\n--- Extracting S2 biomarkers ---")
s2_files = sorted(glob.glob(os.path.join(DATA_DIR, 'S_1*_S2_1_VRG.csv')))
print(f"Found {len(s2_files)} S2 VRG files")

s2_results = []
for fpath in s2_files:
    fname = os.path.basename(fpath).replace('.csv', '')
    pid = int(fname.split('_')[1])
    bio = extract_participant_biomarkers(fpath)
    if bio is not None:
        bio['participant'] = pid
        s2_results.append(bio)

df_s2 = pd.DataFrame(s2_results)
print(f"S2 biomarkers extracted: {len(df_s2)}")

# 4. Match S1 and S2 participants
merged = pd.merge(
    df_s1[['participant', 'cluster_k3']],
    df_s2,
    on='participant',
    how='inner'
)
print(f"Matched participants (S1+S2): {len(merged)}")

# 5. Predict S2 cluster using S1 scaler + S1 K-Means model
for col in feature_cols:
    max_val = merged[col].max()
    merged[col] = merged[col].fillna(max_val)

X_s2 = merged[feature_cols].values
X_s2_scaled = scaler.transform(X_s2)  # S1 기준으로 표준화
s2_predicted = km.predict(X_s2_scaled)

merged['s1_cluster'] = merged['cluster_k3'].values
merged['s2_cluster'] = s2_predicted

# 6. Compute agreement metrics
accuracy = accuracy_score(merged['s1_cluster'], merged['s2_cluster'])
kappa = cohen_kappa_score(merged['s1_cluster'], merged['s2_cluster'])
cm = confusion_matrix(merged['s1_cluster'], merged['s2_cluster'])

print(f"\n{'='*70}")
print(f"  Session Stability Results")
print(f"{'='*70}")
print(f"  Accuracy (S1 == S2): {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"  Cohen's Kappa:       {kappa:.3f}")
print(f"\n  Kappa interpretation:")
print(f"    <0.20 = Poor")
print(f"    0.21-0.40 = Fair")
print(f"    0.41-0.60 = Moderate")
print(f"    0.61-0.80 = Substantial")
print(f"    0.81-1.00 = Almost Perfect")

print(f"\n  Confusion Matrix (rows=S1, cols=S2):")
print(f"  {'':>8} S2_C0  S2_C1  S2_C2")
for i in range(3):
    row = '  '.join(f'{cm[i,j]:>5d}' for j in range(3))
    print(f"  S1_C{i}  {row}")

# Per-cluster stability
print(f"\n  Per-cluster stability:")
for c in range(3):
    mask = merged['s1_cluster'] == c
    n_total = mask.sum()
    n_same = (merged.loc[mask, 's2_cluster'] == c).sum()
    pct = n_same / n_total * 100 if n_total > 0 else 0
    print(f"    C{c}: {n_same}/{n_total} stayed ({pct:.1f}%)")

# 7. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Step 6: Session Stability (S1 vs S2, N={len(merged)})\n'
             f'Accuracy={accuracy:.1%}  |  Cohen\'s Kappa={kappa:.3f}',
             fontsize=14, fontweight='bold', color='white')

# Confusion matrix heatmap
ax = axes[0]
im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels([f'S2 C{i}' for i in range(3)])
ax.set_yticklabels([f'S1 C{i}' for i in range(3)])
ax.set_title('Confusion Matrix (S1 vs S2)', fontweight='bold')
for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
        ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=14, fontweight='bold', color=color)
plt.colorbar(im, ax=ax, shrink=0.8)

# Per-cluster retention rate
ax = axes[1]
retention = []
for c in range(3):
    mask = merged['s1_cluster'] == c
    n_total = mask.sum()
    n_same = (merged.loc[mask, 's2_cluster'] == c).sum()
    retention.append(n_same / n_total * 100 if n_total > 0 else 0)
bars = ax.bar(range(3), retention, color=COLORS[:3], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, retention):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
ax.set_xticks(range(3))
ax.set_xticklabels([f'C{i}' for i in range(3)])
ax.set_ylabel('Retention Rate (%)')
ax.set_title('Per-Cluster Retention (S1 to S2)', fontweight='bold')
ax.set_ylim(0, 105)

# Sankey-style flow (simplified as stacked bars)
ax = axes[2]
flow_data = []
for s1_c in range(3):
    mask = merged['s1_cluster'] == s1_c
    for s2_c in range(3):
        count = (merged.loc[mask, 's2_cluster'] == s2_c).sum()
        flow_data.append({'s1': f'S1_C{s1_c}', 's2': f'S2_C{s2_c}', 'count': count})

df_flow = pd.DataFrame(flow_data)
bottoms = [0, 0, 0]
for s2_c in range(3):
    vals = [df_flow[(df_flow['s1'] == f'S1_C{s1_c}') & (df_flow['s2'] == f'S2_C{s2_c}')]['count'].values[0] for s1_c in range(3)]
    ax.bar(range(3), vals, bottom=bottoms, color=COLORS[s2_c], label=f'-> S2 C{s2_c}',
           edgecolor='white', linewidth=0.3, alpha=0.8)
    bottoms = [b + v for b, v in zip(bottoms, vals)]
ax.set_xticks(range(3))
ax.set_xticklabels([f'S1 C{i}' for i in range(3)])
ax.set_ylabel('Count')
ax.set_title('S1 Cluster -> S2 Cluster Flow', fontweight='bold')
ax.legend(fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(OUT_DIR, 'step6_session_stability.png'), bbox_inches='tight')
plt.close()
print(f"\nSaved: step6_session_stability.png")

# Save results
merged[['participant', 's1_cluster', 's2_cluster']].to_csv(
    os.path.join(OUT_DIR, 'step6_session_labels.csv'), index=False)
print("Saved: step6_session_labels.csv")

print("\nStep 6 complete!")
