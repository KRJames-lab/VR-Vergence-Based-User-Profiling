"""
Figure 1 for IDW paper: cluster-mean vergence step responses (raw degrees).

Reuses the preprocessing / transition-detection of step6_session_stability.py.
For every VRG transition we take the filtered vergence (degrees) on a common
post-transition time grid. Transition DIRECTION is decided by the stimulus
(target depth zT), not by the participant's own response, so that abnormal
clusters are not mis-binned:
    near target (zT < 0.40 m) -> convergence (ideal ~10 deg)
    far  target (zT >= 0.40 m) -> divergence (ideal ~8 deg)
Curves are averaged WITHIN each participant first, then across participants
within each K=3 cluster (mean +/- SE over participants). A dashed line marks
the mean ideal vergence angle (target) for each panel.

Cluster label mapping (cluster_k3, per RESEARCH_DATA_LOOKUP.md 407 canon):
    0 = reach-failure (118), 1 = normal (263), 2 = unstable (26)
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# run from project root regardless of caller's working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Config (constants identical to step6_session_stability.py)
# ------------------------------------------------------------------
OUT_DIR = 'eda_results/eye_tracking/full_407'
DATA_DIR = 'datasets/gazebasevr/extracted_full/data'
CLUSTER_CSV = os.path.join(OUT_DIR, 'pilot_biomarkers_with_clusters.csv')
FIG_OUT = 'my_paper/manuscript/fig_cluster_stepresponse.png'

OUTLIER_THRESHOLD = 50
VERGENCE_VELOCITY_THRESHOLD = 1.0
SAMPLING_RATE = 250
DT_S = 1.0 / SAMPLING_RATE
DT_MS = 1000.0 / SAMPLING_RATE
BUTTER_ORDER = 2
BUTTER_CUTOFF_HZ = 40
NEAR_FAR_SPLIT_M = 0.40        # zT below -> near (converge), above -> far (diverge)

T_MAX_MS = 2000.0                                   # post-transition window shown
t_grid = np.arange(0.0, T_MAX_MS + DT_MS, DT_MS)    # common time base (4 ms)

# normal / over-converging / under-converging  (C0 / C1 / C2 in notebook naming)
LABELS = {1: 'Normal', 0: 'Over-converging', 2: 'Under-converging'}
COLORS = {1: '#1f77b4', 0: '#ff7f0e', 2: '#d62728'}
LINESTYLES = {1: '-', 0: '--', 2: ':'}
ORDER = [1, 0, 2]


# ------------------------------------------------------------------
# Preprocessing (from step6) -- adds vergence_filt + ideal_vergence
# ------------------------------------------------------------------
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['time_s'] = df['n'] / 1000.0
    for col in ['lx', 'ly', 'rx', 'ry']:
        df.loc[df[col].abs() > OUTLIER_THRESHOLD, col] = np.nan
    df[['lx', 'ly', 'rx', 'ry']] = df[['lx', 'ly', 'rx', 'ry']].interpolate(method='linear')
    df = df.dropna(subset=['lx', 'rx', 'zT']).copy()
    df['vergence'] = df['lx'] - df['rx']
    vergence_diff = df['vergence'].diff().abs()
    df.loc[vergence_diff > VERGENCE_VELOCITY_THRESHOLD, 'vergence'] = np.nan
    df['vergence'] = df['vergence'].interpolate(method='linear')
    df = df.dropna(subset=['vergence']).copy()

    b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ, btype='low', fs=SAMPLING_RATE)
    if len(df) > 3 * max(len(b), len(a)):
        df['vergence_filt'] = filtfilt(b, a, df['vergence'].values)
    else:
        df['vergence_filt'] = df['vergence'].values

    ipd = (df['crx'] - df['clx']).median()
    df['ideal_vergence'] = 2 * np.degrees(np.arctan((ipd / 2) / df['zT']))
    return df


def find_transitions(df):
    zT_diff = df['zT'].diff().abs()
    return df.index[zT_diff > 0.01].tolist()


def participant_curves(filepath):
    """Return per-participant mean curves and ideals:
       (near_curve, far_curve, near_ideal, far_ideal); any may be None/nan."""
    try:
        df = load_and_preprocess(filepath)
    except Exception:
        return None, None, np.nan, np.nan
    transitions = find_transitions(df)
    near_traces, far_traces, near_ideals, far_ideals = [], [], [], []
    for i, t_idx in enumerate(transitions):
        next_idx = transitions[i + 1] if i + 1 < len(transitions) else None
        if next_idx is None:
            seg = df.loc[t_idx:].copy()
        else:
            seg = df.loc[t_idx:next_idx - 1].copy()
        if len(seg) < 20:
            continue
        t0 = seg['time_s'].iloc[0]
        t_rel_ms = (seg['time_s'].values - t0) * 1000.0
        v = seg['vergence_filt'].values
        ideal = seg['ideal_vergence'].iloc[-1]
        zt = seg['zT'].iloc[-1]
        v_before = v[0]
        if abs(ideal - v_before) < 0.5:                 # require a real step
            continue
        interp = np.interp(t_grid, t_rel_ms, v, left=np.nan, right=np.nan)
        if zt < NEAR_FAR_SPLIT_M:                        # near target -> convergence
            near_traces.append(interp)
            near_ideals.append(ideal)
        else:                                            # far target -> divergence
            far_traces.append(interp)
            far_ideals.append(ideal)

    near_mean = np.nanmean(np.vstack(near_traces), axis=0) if near_traces else None
    far_mean = np.nanmean(np.vstack(far_traces), axis=0) if far_traces else None
    near_ideal = np.mean(near_ideals) if near_ideals else np.nan
    far_ideal = np.mean(far_ideals) if far_ideals else np.nan
    return near_mean, far_mean, near_ideal, far_ideal


# ------------------------------------------------------------------
# Collect per-participant curves, grouped by cluster
# ------------------------------------------------------------------
def main():
    df_clusters = pd.read_csv(CLUSTER_CSV)
    print(f"Loaded {len(df_clusters)} participants")
    print(f"cluster_k3 distribution: {df_clusters['cluster_k3'].value_counts().to_dict()}")

    store = {c: {'near': [], 'far': [], 'near_ideal': [], 'far_ideal': []} for c in [0, 1, 2]}
    missing = 0
    for _, row in df_clusters.iterrows():
        pid = str(int(row['participant']))
        sess = row['session']
        c = int(row['cluster_k3'])
        matches = glob.glob(os.path.join(DATA_DIR, f'S_{pid}_{sess}_1_VRG.csv'))
        if not matches:
            missing += 1
            continue
        near, far, ni, fi = participant_curves(matches[0])
        if near is not None:
            store[c]['near'].append(near)
            store[c]['near_ideal'].append(ni)
        if far is not None:
            store[c]['far'].append(far)
            store[c]['far_ideal'].append(fi)
    print(f"Missing VRG files: {missing}")

    stats = {}
    for c in [0, 1, 2]:
        stats[c] = {}
        for d in ['near', 'far']:
            stack = np.vstack(store[c][d])
            n_valid = np.sum(~np.isnan(stack), axis=0)
            mean = np.nanmean(stack, axis=0)
            se = np.nanstd(stack, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            stats[c][d] = (mean, se, stack.shape[0])
        stats[c]['near_ideal'] = np.nanmean(store[c]['near_ideal'])
        stats[c]['far_ideal'] = np.nanmean(store[c]['far_ideal'])
        idx_1s = int(1000 / DT_MS)
        print(f"  cluster {c} ({LABELS[c]:<13}) "
              f"n_near={stats[c]['near'][2]:3d} n_far={stats[c]['far'][2]:3d} | "
              f"near@1s={stats[c]['near'][0][idx_1s]:.2f} (ideal {stats[c]['near_ideal']:.2f}) | "
              f"far@1s={stats[c]['far'][0][idx_1s]:.2f} (ideal {stats[c]['far_ideal']:.2f})")

    # signed steady-state error (1.5 s onward) -- sign reveals over/under-convergence
    ss0 = int(1500 / DT_MS)
    print("\nSigned steady-state error (mean vergence - ideal, 1.5-2.0 s):")
    for c in ORDER:
        near_ss = np.nanmean(stats[c]['near'][0][ss0:]) - stats[c]['near_ideal']
        far_ss = np.nanmean(stats[c]['far'][0][ss0:]) - stats[c]['far_ideal']
        print(f"  {LABELS[c]:<13} near={near_ss:+.2f} deg  far={far_ss:+.2f} deg")

    # mean ideal across all clusters for each panel reference line
    near_ref = np.mean([stats[c]['near_ideal'] for c in [0, 1, 2]])
    far_ref = np.mean([stats[c]['far_ideal'] for c in [0, 1, 2]])

    # ------------------------------------------------------------------
    # Plot: 2 panels (near=convergence / far=divergence), 3 cluster means
    # ------------------------------------------------------------------
    plt.rcParams.update({
        'font.size': 9, 'axes.linewidth': 0.6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    panels = [(axes[0], 'near', 'Convergence', near_ref),
              (axes[1], 'far', 'Divergence', far_ref)]
    for ax, d, title, ref in panels:
        ax.axhline(ref, color='0.5', linewidth=0.7, linestyle='--', alpha=0.8,
                   label='Ideal target')
        for c in ORDER:
            mean, se, n = stats[c][d]
            ax.plot(t_grid, mean, color=COLORS[c], linestyle=LINESTYLES[c],
                    linewidth=1.6, label=LABELS[c])
        ax.set_xlabel('Time after transition (ms)')
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, T_MAX_MS)
        ax.grid(True, color='0.92', linewidth=0.5)
    axes[0].set_ylabel('Vergence angle (deg)')
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIG_OUT}")


if __name__ == '__main__':
    main()
