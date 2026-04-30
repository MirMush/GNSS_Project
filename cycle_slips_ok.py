import hatanaka
from rinexreader2 import rinexReader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SatOrbits as so
import datetime as dt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

#load file
filepath = "nuuk1320.24d"

#decompress file in rinex2
#rnx_2 = hatanaka.decompress_on_disk(filepath)

#read file
rnx = rinexReader("data/nuuk1320.24o")
rnx.readFile(
    readConst=["G", "E"],
    sigTypes=["L1", "L2", "L5", "L7", "C1", "P2"]
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

# Galileo: E1 (same as GPS L1) + E5b (L7) or E5a (L5)
GAL_F1  = 1575.42e6   # E1
GAL_F5  = 1176.45e6   # E5a  (RINEX L5)
GAL_F7  = 1207.14e6   # E5b  (RINEX L7)

LAMBDA1 = C_LIGHT / GPS_F1

def _lambdas(svid, f2_sig="L2"):
    """Return (lambda1, lambda2) for constellation + actual second signal used."""
    if svid.startswith("E"):
        f2 = GAL_F7 if f2_sig == "L7" else GAL_F5
        return C_LIGHT / GAL_F1, C_LIGHT / f2
    return C_LIGHT / GPS_F1, C_LIGHT / GPS_F2

def _get_second_phase(sat_data, svid):
    """Return (phase_array, signal_name) for the best available second frequency."""
    if svid.startswith("E"):
        for sig in ["L7", "L5", "L2"]:
            if sig in sat_data.columns and sat_data[sig].notna().sum() > 10:
                return sat_data[sig].values, sig
        return None, None
    # GPS
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

    # mask data gaps > 10 s — large jumps across gaps are not real slips
    dphi[dt > 10] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        dphi = np.where(dt > 0, dphi / dt, np.nan)

    invalid = (dphi == 0) | np.isnan(dphi) | np.isinf(dphi) | np.isnan(tow) | np.isnan(phase)
    keep_idx = ~invalid
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


def identify_cycle_slips_gf(tow, l1, l2, svid="G", sig2="L2"):
    """
    Geometry-Free (GF) cycle slip detection.

    GF = λ1·L1 - λ2·L2  [metres]

    Constellation-aware: GPS uses L1/L2, Galileo uses E1/E5b (different λ2).
    A 1-cycle slip on L1 → jump of λ1; on L2 → jump of λ2.
    Threshold = λ1/2 to catch single-cycle slips.
    """
    lam1, lam2 = _lambdas(svid, sig2)

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


# process all satellites — use GF when L2/E5b available, else fall back to L1/E1-only
results_l1 = {}
results_gf = {}
plot_data_l1 = {}
plot_data_gf = {}

valid_svids = [sv for sv in rnx.obsSvid if len(sv) >= 3 and sv[1:].isdigit()]

for svid in valid_svids:
    print(f"Processing {svid}...")

    sat_data = rnx.get_svid_data(svid, ["L1", "L2", "L5", "L7"])

    if sat_data.empty or 'L1' not in sat_data.columns:
        print(f"  No L1 data — skipping")
        continue

    tow      = (sat_data.index - sat_data.index[0]).total_seconds().values
    phase_l1 = sat_data['L1'].values

    # --- L1/E1-only ---
    slips_l1, t_c, d_c, m_c, res_l1 = identify_cycle_slips(tow, phase_l1)
    results_l1[svid]   = len(slips_l1)
    plot_data_l1[svid] = (t_c, res_l1, slips_l1)

    # --- GF (dual-frequency) — picks best available second signal ---
    phase_l2, sig2 = _get_second_phase(sat_data, svid)
    if phase_l2 is not None:
        lam1_sv, _ = _lambdas(svid, sig2)
        slips_gf, t_gf, d_gf, m_gf, res_gf, lam1_sv = identify_cycle_slips_gf(
            tow, phase_l1, phase_l2, svid, sig2)
        results_gf[svid]   = len(slips_gf)
        plot_data_gf[svid] = (t_gf, res_gf, slips_gf, lam1_sv)
        print(f"  L1: {len(slips_l1)} slips | GF ({sig2}): {len(slips_gf)} slips")
    else:
        print(f"  L1: {len(slips_l1)} slips | GF: no second frequency")

# recap table
print("\n" + "="*40)
print(f"{'SAT':<6}  {'L1 slips':>8}  {'GF slips':>8}")
print("="*40)
for sv in results_l1:
    gf_str = str(results_gf[sv]) if sv in results_gf else "N/A"
    print(f"{sv:<6}  {results_l1[sv]:>8}  {gf_str:>8}")

# --- TIMELINE PLOT ---
# one row per satellite, x = time, red marks = detected slips
all_svids = list(plot_data_l1.keys())
gps_svids = [sv for sv in all_svids if sv.startswith("G")]
gal_svids = [sv for sv in all_svids if sv.startswith("E")]
ordered   = gps_svids + gal_svids

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(ordered) * 0.4)),
                          gridspec_kw={'width_ratios': [1, 1]})

for ax, method, data_dict, title, color_slip in [
    (axes[0], "L1-only",
     {sv: (plot_data_l1[sv][0], plot_data_l1[sv][1], plot_data_l1[sv][2]) for sv in ordered if sv in plot_data_l1},
     "L1-only detection", "red"),
    (axes[1], "GF",
     {sv: (plot_data_gf[sv][0], plot_data_gf[sv][1], plot_data_gf[sv][2]) for sv in ordered if sv in plot_data_gf},
     "Geometry-Free detection", "red"),
]:
    for yi, svid in enumerate(ordered):
        if svid not in data_dict:
            continue
        t_c, _, slips = data_dict[svid]
        if len(t_c) == 0:
            continue
        # visibility bar
        color_sv = 'steelblue' if svid.startswith("G") else 'darkorange'
        ax.plot([t_c[0], t_c[-1]], [yi, yi], color=color_sv, linewidth=1.5, alpha=0.4)
        # slip markers
        if len(slips) > 0:
            ax.scatter(t_c[slips], np.full(len(slips), yi),
                       color=color_slip, s=12, zorder=5)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    # separator between GPS and Galileo
    if gal_svids:
        sep = len(gps_svids) - 0.5
        ax.axhline(sep, color='grey', linestyle=':', linewidth=0.8)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else 0,
                sep - 0.3, "GPS", fontsize=7, color='steelblue')
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else 0,
                sep + 0.8, "Galileo", fontsize=7, color='darkorange')

fig.suptitle("Cycle Slip Timeline — red marks = detected slips", fontsize=11)
plt.tight_layout()
plt.show()

# --- BAR CHART: slip counts per satellite ---
sv_labels = ordered
counts_l1 = [results_l1.get(sv, 0) for sv in sv_labels]
counts_gf = [results_gf.get(sv, 0) for sv in sv_labels]

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


# ------------------------------------------------------------------------------------------------
# USE SP3 FILES
# ------------------------------------------------------------------------------------------------
# read receiver position
def get_receiver_xyz(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            if 'APPROX POSITION XYZ' in line:
                x = float(line[0:14])
                y = float(line[14:28])
                z = float(line[28:42])
                return np.array([x, y, z])
    return None

rx_xyz = get_receiver_xyz("data/nuuk1320.24o")
print(f"Receiver xyx: {rx_xyz}")

# load sp3 file
svpos = so.sp3Orbits("data/COD0MGXFIN_20241320000_01D_05M_ORB.SP3")

# examples to understand
epoch = dt.datetime(2024, 5, 11, 0, 0, 0) # example epoch
satpos = svpos.getSvPos(epoch) #position of all the satellites at that epoch
print(satpos)
sat_xyz = satpos.loc['G05', ['X', 'Y', 'Z']].values # position of a specific satellite
print(f"G05 position: {sat_xyz}")
rho = np.linalg.norm(sat_xyz - rx_xyz) # distance satellite-receiver
print(f"Distance receiver-G05: {rho/1000:.1f} km")


# ------------------------------------------------------------------------------------------------
# PLOT AT 350km
# ------------------------------------------------------------------------------------------------
R_EARTH = 6371000.0      # m
H_IONO = 350000.0        # 350 km
R_IONO = R_EARTH + H_IONO


def ecef_to_latlon(xyz):
    """
    Simple spherical Earth ECEF -> lat/lon.
    Good enough for mapping IPPs visually.
    """
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def ionospheric_pierce_point(rx_xyz, sat_xyz, h_iono=350000.0):
    """
    Find intersection between receiver-satellite line and ionospheric shell.
    Uses spherical Earth approximation.
    """
    R = R_EARTH + h_iono

    u = sat_xyz - rx_xyz
    u = u / np.linalg.norm(u)

    # Solve |rx + t*u|^2 = R^2
    b = 2 * np.dot(rx_xyz, u)
    c = np.dot(rx_xyz, rx_xyz) - R**2

    disc = b**2 - 4*c
    if disc < 0:
        return None

    t1 = (-b + np.sqrt(disc)) / 2
    t2 = (-b - np.sqrt(disc)) / 2

    # Need positive intersection in satellite direction
    t_candidates = [t for t in [t1, t2] if t > 0]
    if not t_candidates:
        return None

    t = min(t_candidates)
    ipp_xyz = rx_xyz + t * u

    return ipp_xyz


slip_points = []

for svid in plot_data_gf:
    t_gf, res_gf, slips, lam1_sv = plot_data_gf[svid]

    if len(slips) == 0:
        continue

    for slip_idx in slips:
        slip_time_seconds = t_gf[slip_idx]

        # Convert relative seconds back to absolute epoch
        epoch = rnx.fileStart + dt.timedelta(seconds=float(slip_time_seconds))

        try:
            satpos = svpos.getSvPos(epoch)

            if svid not in satpos.index:
                continue

            sat_xyz = satpos.loc[svid, ['X', 'Y', 'Z']].values.astype(float)

            ipp_xyz = ionospheric_pierce_point(rx_xyz, sat_xyz, H_IONO)

            if ipp_xyz is None:
                continue

            lat, lon = ecef_to_latlon(ipp_xyz)

            slip_points.append({
                "svid": svid,
                "epoch": epoch,
                "lat": lat,
                "lon": lon,
                "residual": res_gf[slip_idx]
            })

        except Exception as e:
            print(f"Could not process {svid} at {epoch}: {e}")

slip_df = pd.DataFrame(slip_points)
print(slip_df.head())
print(f"Total mapped slips: {len(slip_df)}")


fig = plt.figure(figsize=(9, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-90, -10, 50, 85], crs=ccrs.PlateCarree())  # Greenland + Canada

ax.add_feature(cfeature.LAND, alpha=0.4)
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

if not slip_df.empty:
    sc = ax.scatter(
        slip_df["lon"],
        slip_df["lat"],
        c=np.abs(slip_df["residual"]),
        s=35,
        cmap="Reds",
        edgecolor="black",
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc, ax=ax, label="|GF residual| [m]")

    for _, row in slip_df.iterrows():
        ax.text(
            row["lon"],
            row["lat"],
            row["svid"],
            fontsize=7,
            transform=ccrs.PlateCarree()
        )

ax.set_title("Cycle Slips mapped at ionospheric height 350 km")
ax.gridlines(draw_labels=True)

plt.show()