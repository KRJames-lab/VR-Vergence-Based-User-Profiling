"""
VRG Vergence Biomarker Extraction - Pilot (42 participants)
Step response 분석을 통해 각 참가자의 양안 협응 바이오마커를 추출합니다.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================================
# Configuration
# ============================================================
DATA_DIR = 'datasets/gazebasevr/extracted_full/data'
OUT_DIR = 'eda_results/eye_tracking/full_407'
os.makedirs(OUT_DIR, exist_ok=True)

OUTLIER_THRESHOLD = 50  # degrees
ZT_CHANGE_THRESHOLD = 0.01  # meters

# Biomarker extraction parameters
LATENCY_SIGMA = 2        # vergence가 정상상태에서 2σ 이상 변하면 반응 시작으로 판정
SETTLING_BAND_DEG = 1.0   # ideal_vergence ±1° 이내 = 안착 (변경: ±2% → ±1°)
SETTLING_HOLD_MS = 200    # 안착 판정에 필요한 최소 유지 시간 (ms)
STEADY_STATE_START_MS = 1500  # 전환 후 1.5초 이후를 정상상태 구간으로
SMOOTHING_WINDOW = 13     # 오버슈트 계산용 이동평균 윈도우 (13 samples = 52ms at 250Hz)
VERGENCE_VELOCITY_THRESHOLD = 1.0  # degree per sample (=250 deg/s). 이 이상 변하면 아티팩트
SAMPLING_RATE = 250       # Hz
DT_MS = 1000 / SAMPLING_RATE  # 4ms per sample
DT_S = 1.0 / SAMPLING_RATE   # 0.004s per sample

# Butterworth LP filter (학계 표준: 2차 40Hz)
BUTTER_ORDER = 2
BUTTER_CUTOFF_HZ = 40
VELOCITY_ONSET_THRESHOLD = 1.5  # deg/s, 반응 시작/종료 판정 (Erkelens & Bobier 2020)

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


# ============================================================
# Helper Functions
# ============================================================
def load_and_preprocess(filepath):
    """CSV 로드 + 이상치 제거 + 선형 보간"""
    df = pd.read_csv(filepath)

    # 시간(초) 컬럼 추가
    df['time_s'] = df['n'] / 1000.0

    # 1차 필터: 절대값 이상치 제거 (±50°)
    for col in ['lx', 'ly', 'rx', 'ry']:
        df.loc[df[col].abs() > OUTLIER_THRESHOLD, col] = np.nan

    # 선형 보간
    df[['lx', 'ly', 'rx', 'ry']] = df[['lx', 'ly', 'rx', 'ry']].interpolate(method='linear')

    # 보간 후에도 남은 NaN 제거 (시작/끝부분)
    df = df.dropna(subset=['lx', 'rx', 'zT']).copy()

    # vergence 계산
    df['vergence'] = df['lx'] - df['rx']

    # 2차 필터: 속도 기반 아티팩트 제거 (blink, tracking loss)
    vergence_diff = df['vergence'].diff().abs()
    artifact_mask = vergence_diff > VERGENCE_VELOCITY_THRESHOLD
    n_artifacts = artifact_mask.sum()
    df.loc[artifact_mask, 'vergence'] = np.nan
    df['vergence'] = df['vergence'].interpolate(method='linear')
    df = df.dropna(subset=['vergence']).copy()

    # 3차 필터: Butterworth 2차 40Hz LP 필터 (학계 표준)
    # 위치 데이터에 먼저 필터 적용 → 이후 속도 산출 (표준 순서)
    b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ, btype='low', fs=SAMPLING_RATE)
    if len(df) > 3 * max(len(b), len(a)):  # filtfilt 최소 길이 확인
        df['vergence_filt'] = filtfilt(b, a, df['vergence'].values)
    else:
        df['vergence_filt'] = df['vergence'].values

    # 속도 산출: Two-Point Central Difference (Bahill et al. 1982)
    # v(n) = [x(n+1) - x(n-1)] / (2 * dt)
    verg = df['vergence_filt'].values
    velocity = np.zeros(len(verg))
    velocity[1:-1] = (verg[2:] - verg[:-2]) / (2 * DT_S)
    velocity[0] = (verg[1] - verg[0]) / DT_S
    velocity[-1] = (verg[-1] - verg[-2]) / DT_S
    df['vergence_velocity'] = velocity

    # 개인별 IPD
    ipd = (df['crx'] - df['clx']).median()

    # ideal vergence
    df['ideal_vergence'] = 2 * np.degrees(np.arctan((ipd / 2) / df['zT']))

    # error (필터링된 vergence 기준)
    df['error'] = df['vergence_filt'] - df['ideal_vergence']

    return df, ipd, n_artifacts


def find_transitions(df):
    """zT 값이 변하는 시점(전환점)을 찾습니다."""
    zT_diff = df['zT'].diff().abs()
    transition_indices = df.index[zT_diff > ZT_CHANGE_THRESHOLD].tolist()
    return transition_indices


def extract_step_biomarkers(df, transition_idx, next_transition_idx=None):
    """
    하나의 step transition에서 바이오마커를 추출합니다.

    Parameters:
        df: 전처리된 DataFrame
        transition_idx: 전환이 발생한 인덱스
        next_transition_idx: 다음 전환 인덱스 (세그먼트 끝)

    Returns:
        dict of biomarkers, or None if extraction fails
    """
    if next_transition_idx is None:
        segment = df.loc[transition_idx:].copy()
    else:
        segment = df.loc[transition_idx:next_transition_idx - 1].copy()

    if len(segment) < 20:  # 최소 샘플 수
        return None

    # 시간을 전환 시점 기준으로 0부터 시작
    t0 = segment['time_s'].iloc[0]
    segment['t_rel'] = segment['time_s'] - t0
    segment['t_rel_ms'] = segment['t_rel'] * 1000

    # 이 세그먼트의 ideal vergence (전환 후 값)
    ideal = segment['ideal_vergence'].iloc[-1]

    # 필터링된 vergence와 사전 계산된 velocity 사용
    v_before = segment['vergence_filt'].iloc[0]

    # 목표 변화량
    delta = ideal - v_before
    if abs(delta) < 0.5:
        return None

    direction = 'converge' if delta > 0 else 'diverge'

    vergence = segment['vergence_filt'].values  # Butterworth 필터링된 신호
    velocity = segment['vergence_velocity'].values  # TPCD로 산출된 속도
    t_ms = segment['t_rel_ms'].values

    # ------ 1. Latency (반응 시간) ------
    # Raw vergence(필터링 전)에서 2σ 변화 시점으로 측정.
    # 필터링된 신호는 step change를 smooth하게 만들어 latency가 0에 가까워지므로 부적합.
    raw_vergence = segment['vergence'].values  # 필터링 전 원본
    pre_window = raw_vergence[:max(1, int(50 / DT_MS))]
    pre_mean = np.mean(pre_window)
    pre_std = np.std(pre_window) if len(pre_window) > 1 else 0.5
    threshold = LATENCY_SIGMA * max(pre_std, 0.3)
    latency_ms = np.nan
    for i in range(len(raw_vergence)):
        if abs(raw_vergence[i] - pre_mean) > threshold:
            latency_ms = t_ms[i]
            break

    # ------ 2. Settling Time (정착 시간) ------
    settling_ms = np.nan
    hold_samples = int(SETTLING_HOLD_MS / DT_MS)
    for i in range(len(vergence) - hold_samples):
        window = vergence[i:i + hold_samples]
        if np.all(np.abs(window - ideal) <= SETTLING_BAND_DEG):
            settling_ms = t_ms[i]
            break

    # ------ 3. Overshoot ------
    # Butterworth 필터링된 신호에서 직접 계산 (추가 smoothing 불필요)
    if direction == 'converge':
        peak = np.max(vergence)
        overshoot_abs = peak - ideal
    else:
        peak = np.min(vergence)
        overshoot_abs = ideal - peak
    overshoot_pct = max(0, (overshoot_abs / abs(delta)) * 100) if abs(delta) > 0 else 0

    # ------ 4. Steady-State Error ------
    ss_start_idx = np.searchsorted(t_ms, STEADY_STATE_START_MS)
    if ss_start_idx < len(vergence) - 5:
        ss_vergence = vergence[ss_start_idx:]
        ss_error = np.mean(np.abs(ss_vergence - ideal))
        ss_std = np.std(ss_vergence)
    else:
        ss_error = np.nan
        ss_std = np.nan

    # ------ 5. Jitter ------
    jitter = ss_std

    # ------ 6. Response Speed (deg/s) ------
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

    # ------ 7. Peak Velocity (deg/s) ------
    # onset(1.5 deg/s) ~ offset 구간에서만 max (학계 표준)
    # TPCD로 산출된 velocity에서 transient 구간의 최대 절대값
    onset_idx = None
    offset_idx = None
    for i in range(len(velocity)):
        if abs(velocity[i]) > VELOCITY_ONSET_THRESHOLD:
            if onset_idx is None:
                onset_idx = i
            offset_idx = i  # 마지막으로 threshold 넘은 시점 계속 업데이트

    if onset_idx is not None and offset_idx is not None and offset_idx > onset_idx:
        transient_vel = np.abs(velocity[onset_idx:offset_idx + 1])
        peak_velocity = np.max(transient_vel)
    else:
        peak_velocity = np.max(np.abs(velocity)) if len(velocity) > 0 else np.nan

    return {
        'direction': direction,
        'delta': delta,
        'latency_ms': latency_ms,
        'settling_ms': settling_ms,
        'overshoot_pct': overshoot_pct,
        'ss_error': ss_error,
        'jitter': jitter,
        'speed_deg_s': speed,
        'peak_velocity': peak_velocity,
    }


# ============================================================
# Main: Process all VRG files
# ============================================================
print("=" * 70)
print("  VRG Vergence Biomarker Extraction - Full 502")
print("=" * 70)

# S1만 사용 (S2는 재현성 검증용으로 보존)
vrg_files = sorted(glob.glob(os.path.join(DATA_DIR, 'S_1*_S1_1_VRG.csv')))
print(f"\nFound {len(vrg_files)} VRG S1 files")

all_participant_biomarkers = []

for fpath in vrg_files:
    fname = os.path.basename(fpath).replace('.csv', '')
    parts = fname.split('_')
    pid = parts[1]
    session = parts[2]

    print(f"\n--- Processing {fname} ---")

    # Load and preprocess
    df, ipd, n_artifacts = load_and_preprocess(fpath)
    print(f"  Samples: {len(df)}, IPD: {ipd * 1000:.1f}mm, Artifacts removed: {n_artifacts}")

    # Find transitions
    transitions = find_transitions(df)
    print(f"  Transitions: {len(transitions)}")

    # Extract biomarkers for each step
    step_results = []
    for i, t_idx in enumerate(transitions):
        next_idx = transitions[i + 1] if i + 1 < len(transitions) else None
        bio = extract_step_biomarkers(df, t_idx, next_idx)
        if bio is not None:
            bio['step_num'] = i + 1
            step_results.append(bio)

    if len(step_results) == 0:
        print("  WARNING: No valid steps extracted!")
        continue

    df_steps = pd.DataFrame(step_results)

    # Aggregate per participant (mean of all steps)
    conv = df_steps[df_steps['direction'] == 'converge']
    div = df_steps[df_steps['direction'] == 'diverge']

    participant_bio = {
        'participant': pid,
        'session': session,
        'ipd_mm': ipd * 1000,
        'n_steps': len(df_steps),
        # Overall
        'latency_ms_mean': df_steps['latency_ms'].mean(),
        'latency_ms_std': df_steps['latency_ms'].std(),
        'settling_ms_mean': df_steps['settling_ms'].mean(),
        'settling_ms_std': df_steps['settling_ms'].std(),
        'overshoot_pct_mean': df_steps['overshoot_pct'].mean(),
        'overshoot_pct_std': df_steps['overshoot_pct'].std(),
        'ss_error_mean': df_steps['ss_error'].mean(),
        'ss_error_std': df_steps['ss_error'].std(),
        'jitter_mean': df_steps['jitter'].mean(),
        'jitter_std': df_steps['jitter'].std(),
        'speed_mean': df_steps['speed_deg_s'].mean(),
        'speed_std': df_steps['speed_deg_s'].std(),
        # Convergence only
        'conv_latency_ms': conv['latency_ms'].mean() if len(conv) > 0 else np.nan,
        'conv_settling_ms': conv['settling_ms'].mean() if len(conv) > 0 else np.nan,
        'conv_overshoot_pct': conv['overshoot_pct'].mean() if len(conv) > 0 else np.nan,
        'conv_ss_error': conv['ss_error'].mean() if len(conv) > 0 else np.nan,
        'conv_speed': conv['speed_deg_s'].mean() if len(conv) > 0 else np.nan,
        'conv_peak_velocity': conv['peak_velocity'].mean() if len(conv) > 0 else np.nan,
        # Divergence only
        'div_latency_ms': div['latency_ms'].mean() if len(div) > 0 else np.nan,
        'div_settling_ms': div['settling_ms'].mean() if len(div) > 0 else np.nan,
        'div_overshoot_pct': div['overshoot_pct'].mean() if len(div) > 0 else np.nan,
        'div_ss_error': div['ss_error'].mean() if len(div) > 0 else np.nan,
        'div_speed': div['speed_deg_s'].mean() if len(div) > 0 else np.nan,
        'div_peak_velocity': div['peak_velocity'].mean() if len(div) > 0 else np.nan,
        # MAE overall
        'vergence_mae': np.abs(df['error']).mean(),
        'vergence_rmse': np.sqrt((df['error'] ** 2).mean()),
    }

    all_participant_biomarkers.append(participant_bio)

    print(f"  Latency: {participant_bio['latency_ms_mean']:.0f}ms | "
          f"Settling: {participant_bio['settling_ms_mean']:.0f}ms | "
          f"Overshoot: {participant_bio['overshoot_pct_mean']:.1f}% | "
          f"SS Error: {participant_bio['ss_error_mean']:.2f}° | "
          f"Jitter: {participant_bio['jitter_mean']:.2f}° | "
          f"MAE: {participant_bio['vergence_mae']:.2f}°")

# ============================================================
# Save results
# ============================================================
df_bio = pd.DataFrame(all_participant_biomarkers)
csv_path = os.path.join(OUT_DIR, 'pilot_biomarkers.csv')
df_bio.to_csv(csv_path, index=False)
print(f"\n\nSaved biomarkers: {csv_path}")
print(f"Total participants: {len(df_bio)}")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "=" * 70)
print("  Summary Statistics (All Participants)")
print("=" * 70)

cols_to_summarize = [
    'latency_ms_mean', 'settling_ms_mean', 'overshoot_pct_mean',
    'ss_error_mean', 'jitter_mean', 'speed_mean', 'vergence_mae', 'vergence_rmse'
]
labels = [
    'Latency (ms)', 'Settling (ms)', 'Overshoot (%)',
    'SS Error (°)', 'Jitter (°)', 'Speed (°/s)', 'MAE (°)', 'RMSE (°)'
]

for col, label in zip(cols_to_summarize, labels):
    vals = df_bio[col].dropna()
    print(f"  {label:<20}: mean={vals.mean():8.2f}  std={vals.std():8.2f}  "
          f"min={vals.min():8.2f}  max={vals.max():8.2f}  "
          f"P25={vals.quantile(0.25):8.2f}  P75={vals.quantile(0.75):8.2f}")

# ============================================================
# Visualization: Biomarker distributions
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('VRG Biomarker Distributions (Pilot, N={})'.format(len(df_bio)),
             fontsize=16, fontweight='bold', color='white')

colors = ['#e94560', '#00b4d8', '#2a9d8f', '#e9c46a', '#533483', '#f4a261', '#58a6ff', '#f85149']

for idx, (col, label, color) in enumerate(zip(cols_to_summarize, labels, colors)):
    ax = axes[idx // 4][idx % 4]
    vals = df_bio[col].dropna()
    ax.hist(vals, bins=15, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(vals.mean(), color='white', linestyle='--', linewidth=1.5, label=f'Mean: {vals.mean():.1f}')
    ax.axvline(vals.median(), color='#3fb950', linestyle=':', linewidth=1.5, label=f'Median: {vals.median():.1f}')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, 'pilot_biomarker_distributions.png'),
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"\nSaved: pilot_biomarker_distributions.png")

# ============================================================
# Visualization: Convergence vs Divergence comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Convergence vs Divergence Biomarkers', fontsize=14, fontweight='bold', color='white')

pairs = [
    ('conv_latency_ms', 'div_latency_ms', 'Latency (ms)'),
    ('conv_settling_ms', 'div_settling_ms', 'Settling Time (ms)'),
    ('conv_overshoot_pct', 'div_overshoot_pct', 'Overshoot (%)'),
]

for idx, (c_col, d_col, label) in enumerate(pairs):
    ax = axes[idx]
    c_vals = df_bio[c_col].dropna()
    d_vals = df_bio[d_col].dropna()
    positions = [1, 2]
    bp = ax.boxplot([c_vals, d_vals], positions=positions, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor='#161b22', edgecolor='#58a6ff'),
                    medianprops=dict(color='#3fb950', linewidth=2),
                    whiskerprops=dict(color='#8b949e'),
                    capprops=dict(color='#8b949e'),
                    flierprops=dict(marker='o', markerfacecolor='#f85149', markersize=4))
    bp['boxes'][0].set_facecolor('#1a3a5c')
    bp['boxes'][1].set_facecolor('#3a1a3c')
    ax.set_xticklabels(['Converge', 'Diverge'])
    ax.set_title(label, fontsize=11, fontweight='bold')

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(OUT_DIR, 'pilot_conv_vs_div.png'),
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved: pilot_conv_vs_div.png")

print("\nPilot biomarker extraction complete!")
