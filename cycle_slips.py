import hatanaka
from rinexreader2 import rinexReader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load file
filepath = "nuuk1320.24d"

#decompress file in rinex2
#rnx_2 = hatanaka.decompress_on_disk(filepath)

#read file
rnx = rinexReader("data/nuuk1320.24o")
rnx.readFile(
    readConst=["G", "E"],
    sigTypes=["L1", "L2", "C1", "P2"]
)

print("Start:", rnx.fileStart)
print("End:", rnx.fileEnd)
print("Epochs:", len(rnx.obs))
print("Satellites:", len(rnx.obsSvid))

g05 = rnx.get_svid_data("G05", ["L1", "L2"])
print(g05.head())

# Signal constants
C_LIGHT = 299792458.0  # m/s

# GPS L1/L2
GPS_F1 = 1575.42e6
GPS_F2 = 1227.60e6

# Galileo E1/E5b  (RINEX 2 maps Galileo L1→E1, L2→E5b)
GAL_F1 = 1575.42e6
GAL_F2 = 1207.14e6

LAMBDA1 = C_LIGHT / GPS_F1  # GPS L1 wavelength, used as fallback for plots

def _lambdas(svid):
    """Return (lambda1, lambda2) in metres for the given constellation."""
    if svid.startswith("E"):
        return C_LIGHT / GAL_F1, C_LIGHT / GAL_F2
    return C_LIGHT / GPS_F1, C_LIGHT / GPS_F2

def movmedian(data, window):
    return pd.Series(data).rolling(window=int(window), center=True, min_periods=1).median().values

def identify_cycle_slips(tow, phase):
    _empty = np.array([])
    if len(tow) < 2:
        return _empty, tow, _empty, _empty, _empty

    dphi = np.diff(phase)
    dphi = np.insert(dphi, 0, dphi[0] if len(dphi) > 0 else 0)

    dt = np.diff(tow)
    dt = np.insert(dt, 0, dt[0])
    dphi = dphi / dt

    repeated_or_nan = (dphi==0) | np.isnan(dphi) | np.isnan(tow) | np.isnan(phase)
    keep_idx = ~repeated_or_nan
    t_clean = tow[keep_idx]
    dphi_clean = dphi[keep_idx]

    if len(dphi_clean) < 2:
        return _empty, t_clean, dphi_clean, dphi_clean, dphi_clean

    medwidth = 60
    mphi = movmedian(dphi_clean, medwidth)
    residual = dphi_clean - mphi

    slip_thresh = 1.0
    ind_slips = np.where(np.abs(residual) > slip_thresh)[0]

    return ind_slips, t_clean, dphi_clean, mphi, residual


def identify_cycle_slips_gf(tow, l1, l2, svid="G"):
    """
    Geometry-Free (GF) cycle slip detection.

    GF = λ1·L1 - λ2·L2  [metres]

    Constellation-aware: GPS uses L1/L2, Galileo uses E1/E5b (different λ2).
    A 1-cycle slip on L1 → jump of λ1; on L2 → jump of λ2.
    Threshold = λ1/2 to catch single-cycle slips.
    """
    lam1, lam2 = _lambdas(svid)

    _empty = np.array([])
    valid = ~(np.isnan(l1) | np.isnan(l2) | np.isnan(tow))
    tow = tow[valid];  l1 = l1[valid];  l2 = l2[valid]

    if len(tow) < 2:
        return _empty, tow, _empty, _empty, _empty

    gf = lam1 * l1 - lam2 * l2

    dgf = np.diff(gf)
    dgf = np.insert(dgf, 0, dgf[0])

    dt = np.diff(tow)
    dt = np.insert(dt, 0, dt[0])
    dgf[dt > 10] = np.nan   # mask data gaps > 10 s

    invalid = (dgf == 0) | np.isnan(dgf)
    keep = ~invalid
    t_clean   = tow[keep]
    dgf_clean = dgf[keep]

    if len(dgf_clean) < 2:
        return _empty, t_clean, dgf_clean, dgf_clean, dgf_clean

    medwidth = 60
    mgf = movmedian(dgf_clean, medwidth)
    residual = dgf_clean - mgf

    slip_thresh = lam1 / 2
    ind_slips = np.where(np.abs(residual) > slip_thresh)[0]

    return ind_slips, t_clean, dgf_clean, mgf, residual, lam1


# process all satellites — use GF when L2 available, else fall back to L1-only
results_l1 = {}
results_gf = {}
plot_data_l1 = {}
plot_data_gf = {}

for svid in rnx.obsSvid:
    print(f"Processing {svid}...")

    sat_data = rnx.get_svid_data(svid, ["L1", "L2"])

    if sat_data.empty or 'L1' not in sat_data.columns:
        print(f"  No L1 data — skipping")
        continue

    tow      = (sat_data.index - sat_data.index[0]).total_seconds().values
    phase_l1 = sat_data['L1'].values

    # --- L1-only ---
    slips_l1, t_c, d_c, m_c, res_l1 = identify_cycle_slips(tow, phase_l1)
    results_l1[svid]  = len(slips_l1)
    plot_data_l1[svid] = (t_c, res_l1, slips_l1)

    # --- GF (dual-frequency) ---
    if 'L2' in sat_data.columns and sat_data['L2'].notna().sum() > 10:
        phase_l2 = sat_data['L2'].values
        slips_gf, t_gf, d_gf, m_gf, res_gf, lam1_sv = identify_cycle_slips_gf(tow, phase_l1, phase_l2, svid)
        results_gf[svid]   = len(slips_gf)
        plot_data_gf[svid] = (t_gf, res_gf, slips_gf, lam1_sv)
        print(f"  L1-only: {len(slips_l1)} slips | GF: {len(slips_gf)} slips")
    else:
        print(f"  L1-only: {len(slips_l1)} slips | GF: no L2")

# recap
print("\n" + "="*40)
print(f"{'SAT':<6}  {'L1 slips':>8}  {'GF slips':>8}")
print("="*40)
for sv in results_l1:
    gf_str = str(results_gf[sv]) if sv in results_gf else "N/A"
    print(f"{sv:<6}  {results_l1[sv]:>8}  {gf_str:>8}")

# --- subplot grid: L1 vs GF side-by-side for each satellite ---
common_svids = [sv for sv in plot_data_l1 if sv in plot_data_gf]
n    = len(common_svids)
ncols = 4
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows * 2, ncols,
                          figsize=(ncols * 4, nrows * 6),
                          sharey=False)
axes = np.array(axes).reshape(nrows * 2, ncols)

for i, svid in enumerate(common_svids):
    row_l1 = (i // ncols) * 2
    row_gf = row_l1 + 1
    col    = i % ncols

    # --- L1 panel ---
    t_c, res, slips = plot_data_l1[svid]
    ax = axes[row_l1, col]
    ax.scatter(t_c, res, s=2, color='black')
    if len(slips) > 0:
        ax.scatter(t_c[slips], res[slips], s=15, color='red', zorder=5,
                   label=f'{len(slips)} slips')
        ax.legend(fontsize=6, loc='upper right')
    ax.axhline(y= 1.0, color='r', linestyle='--', linewidth=0.7)
    ax.axhline(y=-1.0, color='r', linestyle='--', linewidth=0.7)
    ax.set_title(f"{svid} — L1", fontsize=8)
    ax.set_ylim([-5, 5])
    ax.set_ylabel("cycles/s", fontsize=7)
    ax.tick_params(labelsize=6)

    # --- GF panel ---
    t_gf, res_gf, slips_gf, lam1_sv = plot_data_gf[svid]
    ax2 = axes[row_gf, col]
    ax2.scatter(t_gf, res_gf, s=2, color='steelblue')
    if len(slips_gf) > 0:
        ax2.scatter(t_gf[slips_gf], res_gf[slips_gf], s=15, color='red', zorder=5,
                    label=f'{len(slips_gf)} slips')
        ax2.legend(fontsize=6, loc='upper right')
    ax2.axhline(y= lam1_sv/2, color='r', linestyle='--', linewidth=0.7)
    ax2.axhline(y=-lam1_sv/2, color='r', linestyle='--', linewidth=0.7)
    ax2.set_title(f"{svid} — GF", fontsize=8)
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_ylabel("metres", fontsize=7)
    ax2.tick_params(labelsize=6)

# hide unused panels
total_panels = nrows * 2 * ncols
for j in range(i + 1, ncols * nrows):
    axes[(j // ncols) * 2,     j % ncols].set_visible(False)
    axes[(j // ncols) * 2 + 1, j % ncols].set_visible(False)

fig.supxlabel("Time (s)", fontsize=10)
fig.suptitle("Cycle Slip Detection — L1-only (black) vs Geometry-Free (blue)", fontsize=11)
plt.tight_layout()
plt.show()

# --- summary bar chart: L1 vs GF ---
sv_labels   = list(results_l1.keys())
counts_l1   = [results_l1[sv] for sv in sv_labels]
counts_gf   = [results_gf.get(sv, 0) for sv in sv_labels]

x = np.arange(len(sv_labels))
width = 0.4

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(x - width/2, counts_l1, width, label='L1-only', color='steelblue')
ax.bar(x + width/2, counts_gf, width, label='GF (dual-freq)', color='tomato')
ax.set_xticks(x)
ax.set_xticklabels(sv_labels, rotation=45, ha='right')
ax.set_ylabel("Number of cycle slips")
ax.set_title("Cycle Slips per Satellite: L1-only vs Geometry-Free")
ax.legend()
plt.tight_layout()
plt.show()