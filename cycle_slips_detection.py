import hatanaka
from rinexreader2 import rinexReader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# rnx_2 = hatanaka.decompress_on_disk("nuuk1320.24d")

rnx = rinexReader("data/nuuk1320.24o")
rnx.readFile(
    readConst=["G", "E"],
    sigTypes=["L1", "L2", "L5", "L7", "C1", "P2"]
)

print("Start:", rnx.fileStart)
print("End:",   rnx.fileEnd)
print("Epochs:", len(rnx.obs))
print("Satellites:", len(rnx.obsSvid))

# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------
C_LIGHT = 299792458.0

GPS_F1 = 1575.42e6
GPS_F2 = 1227.60e6

GAL_F1 = 1575.42e6   # E1
GAL_F5 = 1176.45e6   # E5a  (RINEX L5)
GAL_F7 = 1207.14e6   # E5b  (RINEX L7)

LAMBDA1 = C_LIGHT / GPS_F1


def _lambdas(svid, f2_sig="L2"):
    if svid.startswith("E"):
        f2 = GAL_F7 if f2_sig == "L7" else GAL_F5
        return C_LIGHT / GAL_F1, C_LIGHT / f2
    return C_LIGHT / GPS_F1, C_LIGHT / GPS_F2


def _get_second_phase(sat_data, svid):
    if svid.startswith("E"):
        for sig in ["L7", "L5", "L2"]:
            if sig in sat_data.columns and sat_data[sig].notna().sum() > 10:
                return sat_data[sig].values, sig
        return None, None
    if "L2" in sat_data.columns and sat_data["L2"].notna().sum() > 10:
        return sat_data["L2"].values, "L2"
    return None, None


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

    dphi[dt > 10] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        dphi = np.where(dt > 0, dphi / dt, np.nan)

    invalid = (dphi == 0) | np.isnan(dphi) | np.isinf(dphi) | np.isnan(tow) | np.isnan(phase)
    t_clean    = tow[~invalid]
    dphi_clean = dphi[~invalid]

    if len(dphi_clean) < 2:
        return _empty, t_clean, dphi_clean, dphi_clean, dphi_clean

    mphi     = movmedian(dphi_clean, 60)
    residual = dphi_clean - mphi
    ind_slips = np.where(np.abs(residual) > 1.0)[0]

    return ind_slips, t_clean, dphi_clean, mphi, residual


def identify_cycle_slips_gf(tow, l1, l2, svid="G", sig2="L2"):
    """
    Geometry-Free (GF) combination: GF = λ1·L1 - λ2·L2  [metres].
    Removes geometry and clock errors; slow ionospheric change remains.
    A 1-cycle slip on L1 causes a jump of λ1; threshold = λ1/2.
    """
    lam1, lam2 = _lambdas(svid, sig2)

    _empty = np.array([])
    valid = ~(np.isnan(l1) | np.isnan(l2) | np.isnan(tow))
    tow = tow[valid];  l1 = l1[valid];  l2 = l2[valid]

    if len(tow) < 2:
        return _empty, tow, _empty, _empty, _empty, lam1

    gf  = lam1 * l1 - lam2 * l2
    dgf = np.diff(gf)
    dgf = np.insert(dgf, 0, dgf[0])

    dt = np.diff(tow)
    dt = np.insert(dt, 0, dt[0])
    dgf[dt > 10] = np.nan

    invalid   = (dgf == 0) | np.isnan(dgf)
    t_clean   = tow[~invalid]
    dgf_clean = dgf[~invalid]

    if len(dgf_clean) < 2:
        return _empty, t_clean, dgf_clean, dgf_clean, dgf_clean, lam1

    mgf      = movmedian(dgf_clean, 60)
    residual = dgf_clean - mgf
    ind_slips = np.where(np.abs(residual) > lam1 / 2)[0]

    return ind_slips, t_clean, dgf_clean, mgf, residual, lam1


# ---------------------------------------------------------------------------
# Process all satellites
# ---------------------------------------------------------------------------
results_l1  = {}
results_gf  = {}
plot_data_l1 = {}
plot_data_gf = {}

valid_svids = [sv for sv in rnx.obsSvid if len(sv) >= 3 and sv[1:].isdigit()]

for svid in valid_svids:
    print(f"Processing {svid}...")

    sat_data = rnx.get_svid_data(svid, ["L1", "L2", "L5", "L7"])

    if sat_data.empty or "L1" not in sat_data.columns:
        print("  No L1 data — skipping")
        continue

    tow      = (sat_data.index - sat_data.index[0]).total_seconds().values
    phase_l1 = sat_data["L1"].values

    slips_l1, t_c, d_c, m_c, res_l1 = identify_cycle_slips(tow, phase_l1)
    results_l1[svid]   = len(slips_l1)
    plot_data_l1[svid] = (t_c, res_l1, slips_l1)

    phase_l2, sig2 = _get_second_phase(sat_data, svid)
    if phase_l2 is not None:
        slips_gf, t_gf, d_gf, m_gf, res_gf, lam1_sv = identify_cycle_slips_gf(
            tow, phase_l1, phase_l2, svid, sig2)
        results_gf[svid]   = len(slips_gf)
        plot_data_gf[svid] = (t_gf, res_gf, slips_gf, lam1_sv)
        print(f"  L1: {len(slips_l1)} | GF ({sig2}): {len(slips_gf)}")
    else:
        print(f"  L1: {len(slips_l1)} | GF: no second frequency")

# ---------------------------------------------------------------------------
# Save results for mapping script
# ---------------------------------------------------------------------------
with open("data/slip_results.pkl", "wb") as f:
    pickle.dump({
        "plot_data_gf": plot_data_gf,
        "file_start":   rnx.fileStart,
    }, f)
print("Saved slip results to data/slip_results.pkl")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 40)
print(f"{'SAT':<6}  {'L1 slips':>8}  {'GF slips':>8}")
print("=" * 40)
for sv in results_l1:
    gf_str = str(results_gf[sv]) if sv in results_gf else "N/A"
    print(f"{sv:<6}  {results_l1[sv]:>8}  {gf_str:>8}")

# ---------------------------------------------------------------------------
# Timeline plot (L1-only | GF side by side)
# ---------------------------------------------------------------------------
all_svids = list(plot_data_l1.keys())
gps_svids = [sv for sv in all_svids if sv.startswith("G")]
gal_svids = [sv for sv in all_svids if sv.startswith("E")]
ordered   = gps_svids + gal_svids

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(ordered) * 0.4)),
                         gridspec_kw={"width_ratios": [1, 1]})

for ax, data_dict, title in [
    (axes[0],
     {sv: plot_data_l1[sv] for sv in ordered if sv in plot_data_l1},
     "L1-only detection"),
    (axes[1],
     {sv: (plot_data_gf[sv][0], plot_data_gf[sv][1], plot_data_gf[sv][2])
      for sv in ordered if sv in plot_data_gf},
     "Geometry-Free detection"),
]:
    for yi, svid in enumerate(ordered):
        if svid not in data_dict:
            continue
        t_c, _, slips = data_dict[svid]
        if len(t_c) == 0:
            continue
        color_sv = "steelblue" if svid.startswith("G") else "darkorange"
        ax.plot([t_c[0], t_c[-1]], [yi, yi], color=color_sv, linewidth=1.5, alpha=0.4)
        if len(slips) > 0:
            ax.scatter(t_c[slips], np.full(len(slips), yi), color="red", s=12, zorder=5)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    if gal_svids:
        sep = len(gps_svids) - 0.5
        ax.axhline(sep, color="grey", linestyle=":", linewidth=0.8)

fig.suptitle("Cycle Slip Timeline — red marks = detected slips", fontsize=11)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Bar chart: slip counts per satellite
# ---------------------------------------------------------------------------
counts_l1 = [results_l1.get(sv, 0) for sv in ordered]
counts_gf = [results_gf.get(sv, 0) for sv in ordered]
x = np.arange(len(ordered))
width = 0.4

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(x - width / 2, counts_l1, width, label="L1-only",        color="steelblue")
ax.bar(x + width / 2, counts_gf, width, label="GF (dual-freq)", color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(ordered, rotation=45, ha="right")
ax.set_ylabel("Number of cycle slips")
ax.set_title("Cycle Slips per Satellite: L1-only vs Geometry-Free")
ax.legend()
plt.tight_layout()
plt.show()
