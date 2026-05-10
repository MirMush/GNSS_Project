import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import SatOrbits as so
import datetime as dt
import pickle
import glob
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIME_WINDOW_SEC = 3600
GRID_DLAT = 1.0
GRID_DLON = 1.0
H_IONO = 350000.0
R_EARTH = 6371000.0
OBS_SAMPLE_SEC = 300

# ---------------------------------------------------------------------------
# SP3 orbits
# ---------------------------------------------------------------------------
svpos = so.sp3Orbits("data/COD0MGXFIN_20241320000_01D_05M_ORB.SP3")

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def get_receiver_xyz(filepath):
    with open(filepath, "r") as f:
        for line in f:
            if "APPROX POSITION XYZ" in line:
                x = float(line[0:14])
                y = float(line[14:28])
                z = float(line[28:42])
                return np.array([x, y, z])
    return None


def ecef_to_latlon(xyz):
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    return np.degrees(np.arcsin(z / r)), np.degrees(np.arctan2(y, x))


def ionospheric_pierce_point(rx_xyz, sat_xyz, h_iono=H_IONO):
    R = R_EARTH + h_iono
    u = sat_xyz - rx_xyz
    u = u / np.linalg.norm(u)
    b = 2 * np.dot(rx_xyz, u)
    c = np.dot(rx_xyz, rx_xyz) - R ** 2
    disc = b ** 2 - 4 * c
    if disc < 0:
        return None
    candidates = [t for t in [(-b + np.sqrt(disc)) / 2,
                               (-b - np.sqrt(disc)) / 2] if t > 0]
    if not candidates:
        return None
    return rx_xyz + min(candidates) * u


def elevation_angle(rx_xyz, sat_xyz):
    rx_up = rx_xyz / np.linalg.norm(rx_xyz)
    los = sat_xyz - rx_xyz
    los = los / np.linalg.norm(los)
    return np.degrees(np.arcsin(np.dot(rx_up, los)))


def azimuth_angle(rx_xyz, sat_xyz):
    lat_r = np.arcsin(rx_xyz[2] / np.linalg.norm(rx_xyz))
    lon_r = np.arctan2(rx_xyz[1], rx_xyz[0])
    N = np.array([-np.sin(lat_r)*np.cos(lon_r),
                  -np.sin(lat_r)*np.sin(lon_r),
                   np.cos(lat_r)])
    E = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
    los = sat_xyz - rx_xyz
    return np.degrees(np.arctan2(np.dot(los, E), np.dot(los, N))) % 360


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def build_grid(lat_min=50, lat_max=90, lon_min=-180, lon_max=180,
               dlat=GRID_DLAT, dlon=GRID_DLON):
    lat_edges = np.arange(lat_min, lat_max + dlat, dlat)
    lon_edges = np.arange(lon_min, lon_max + dlon, dlon)
    return lat_edges, lon_edges


def bin_slips(df, value_col, lat_edges, lon_edges, aggregator="sum"):
    n_lat = len(lat_edges) - 1
    n_lon = len(lon_edges) - 1
    grid = np.full((n_lat, n_lon), np.nan)
    if df.empty:
        return grid
    lat_idx = np.digitize(df["lat"].values, lat_edges) - 1
    lon_idx = np.digitize(df["lon"].values, lon_edges) - 1
    mask = ((lat_idx >= 0) & (lat_idx < n_lat) &
            (lon_idx >= 0) & (lon_idx < n_lon))
    df = df[mask].copy()
    lat_idx = lat_idx[mask]
    lon_idx = lon_idx[mask]
    for li in range(n_lat):
        for lj in range(n_lon):
            sel = (lat_idx == li) & (lon_idx == lj)
            if sel.sum() == 0:
                continue
            vals = df[value_col].values[sel]
            grid[li, lj] = vals.sum() if aggregator == "sum" else np.nanmean(vals)
    return grid


def plot_binned_grid(ax, grid, lon_edges, lat_edges, cmap, norm):
    return ax.pcolormesh(lon_edges, lat_edges, grid,
                         cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), shading="flat")


# ---------------------------------------------------------------------------
# Map functions
# ---------------------------------------------------------------------------
POLAR_PROJ = ccrs.NorthPolarStereo(central_longitude=-40.0)
DATA_CRS = ccrs.PlateCarree()


def make_polar_ax(fig, position=111, full=False):
    ax = fig.add_subplot(position, projection=POLAR_PROJ)
    _add_features(ax, full)
    return ax


def add_polar_ax_to_figure(fig, nrows, ncols, index, full=False):
    ax = fig.add_subplot(nrows, ncols, index, projection=POLAR_PROJ)
    if full:
        ax.set_extent([-180, 180, 50, 90], crs=DATA_CRS)
    else:
        ax.set_extent([-80, 10, 58, 85], crs=DATA_CRS)
    ax.add_feature(cfeature.LAND, alpha=0.4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return ax


def _add_features(ax, full=False):
    ext = [-180, 180, 50, 90] if full else [-80, 10, 58, 85]
    ax.set_extent(ext, crs=DATA_CRS)
    ax.add_feature(cfeature.LAND, alpha=0.4)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="grey", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return ax

# ---------------------------------------------------------------------------
# Skyplot single station
# ---------------------------------------------------------------------------
def make_single_station_skyplot(station, rx_xyz, svids,
                                file_start, file_end, plot_data_gf,
                                svpos, out_dir,
                                sample_sec=OBS_SAMPLE_SEC, elev_cut=15):
    """
    svids        : ALL satellite IDs to plot tracks for (GPS + Galileo).
    plot_data_gf : raw dict from pkl  {svid: (t_gf, res_gf, slips, lam1)}.
    Slip positions are resolved directly from SP3 so every detected slip
    above elev_cut is shown, not just those that survived IPP mapping.
    """
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 85])
    ax.set_yticklabels(["90°", "75°", "60°", "45°", "30°", "15°", "5°"], fontsize=7)
    ax.set_rlabel_position(135)

    obs_sec   = (file_end - file_start).total_seconds()
    t_samples = np.arange(0, obs_sec + sample_sec, sample_sec)

    # assign one colour per satellite (GPS and Galileo use separate palettes)
    gps_svids_sorted = sorted(sv for sv in svids if sv.startswith("G"))
    gal_svids_sorted = sorted(sv for sv in svids if sv.startswith("E"))
    gps_colors = plt.cm.tab20(np.linspace(0, 1, max(len(gps_svids_sorted), 1)))
    gal_colors = plt.cm.tab20b(np.linspace(0, 1, max(len(gal_svids_sorted), 1)))
    sv_color = {}
    for i, sv in enumerate(gps_svids_sorted):
        sv_color[sv] = gps_colors[i]
    for i, sv in enumerate(gal_svids_sorted):
        sv_color[sv] = gal_colors[i]

    legend_handles = []

    for svid in svids:
        style = "-" if svid.startswith("G") else "--"
        color = sv_color.get(svid, "grey")

        # satellite track
        azs, els = [], []
        for t_sec in t_samples:
            epoch = file_start + dt.timedelta(seconds=float(t_sec))
            try:
                satpos = svpos.getSvPos(epoch, const=["G", "E"])
                if svid not in satpos.index:
                    continue
                sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                elev = elevation_angle(rx_xyz, sat_xyz)
                if elev < elev_cut:
                    continue
                az = azimuth_angle(rx_xyz, sat_xyz)
                azs.append(np.radians(az))
                els.append(90 - elev)
            except Exception:
                continue

        if azs:
            ax.plot(azs, els, color=color, linewidth=1.2,
                    alpha=0.75, linestyle=style)
            mid = len(azs) // 2
            ax.text(azs[mid], els[mid], svid,
                    fontsize=5, color=color, ha="center", va="center",
                    fontweight="bold", alpha=0.9)

        # slip positions resolved fresh from SP3 (no pre-filtering)
        if svid not in plot_data_gf:
            continue
        t_gf, _, slips, _ = plot_data_gf[svid]
        slip_az_sv, slip_el_sv = [], []
        for slip_idx in slips:
            t_sec = float(t_gf[slip_idx])
            epoch = file_start + dt.timedelta(seconds=t_sec)
            try:
                satpos = svpos.getSvPos(epoch, const=["G", "E"])
                if svid not in satpos.index:
                    continue
                sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                elev = elevation_angle(rx_xyz, sat_xyz)
                if elev < elev_cut:
                    continue
                az = azimuth_angle(rx_xyz, sat_xyz)
                slip_az_sv.append(np.radians(az))
                slip_el_sv.append(90 - elev)
            except Exception:
                continue

        if slip_az_sv:
            ax.scatter(slip_az_sv, slip_el_sv,
                       color="red", s=30, zorder=8, alpha=0.9,
                       marker="o", edgecolors="darkred", linewidths=0.4)

        # legend entry only for satellites that have a visible track or slips
        if azs or slip_az_sv:
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=1.5,
                       linestyle=style, label=svid)
            )

    # separator entry between GPS and Galileo in legend
    if gps_svids_sorted and gal_svids_sorted:
        gps_handles = [h for h in legend_handles if h.get_label().startswith("G")]
        gal_handles = [h for h in legend_handles if h.get_label().startswith("E")]
        all_handles = (gps_handles
                       + [Line2D([0], [0], color="none", label="- Galileo -")]
                       + gal_handles
                       + [Line2D([0], [0], color="red", marker="o",
                                 linestyle="None", markersize=5,
                                 markeredgecolor="darkred",
                                 label="Cycle slip")])
    else:
        all_handles = legend_handles

    ax.legend(handles=all_handles,
              loc="upper right", bbox_to_anchor=(1.45, 1.12),
              fontsize=6, title="Satellite", title_fontsize=7,
              ncol=2)

    ax.set_title(f"Skyplot: {station.upper()}\n"
                 f"  elev ≥ {elev_cut}°",
                 pad=20, fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, f"{station}_skyplot.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    n_slips = sum(
        len(plot_data_gf[sv][2]) for sv in svids if sv in plot_data_gf
    )
    print(f" Saved skyplot: {out}  ({n_slips} GF slips total)")


# ---------------------------------------------------------------------------
# n_obs builder
# ---------------------------------------------------------------------------
def build_ipp_obs(svids, file_start, file_end, rx_xyz,
                  sample_sec=OBS_SAMPLE_SEC):
    obs_sec = (file_end - file_start).total_seconds()
    t_samples = np.arange(0, obs_sec + sample_sec, sample_sec)
    rows = []
    for t_sec in t_samples:
        epoch = file_start + dt.timedelta(seconds=float(t_sec))
        tw_idx = int(t_sec // TIME_WINDOW_SEC)
        try:
            satpos = svpos.getSvPos(epoch, const=['G', 'E'])
        except Exception:
            continue
        for svid in svids:
            if svid not in satpos.index:
                continue
            try:
                sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                if elevation_angle(rx_xyz, sat_xyz) < 15:
                    continue
                ipp = ionospheric_pierce_point(rx_xyz, sat_xyz)
                if ipp is None:
                    continue
                lat, lon = ecef_to_latlon(ipp)
                rows.append({"lat": lat, "lon": lon,
                             "time_window": tw_idx, "n_obs": 1})
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Skyplot all stations together
# ---------------------------------------------------------------------------
def make_combined_skyplot(all_station_data, svpos, out_dir,
                          sample_sec=OBS_SAMPLE_SEC, elev_cut=15):
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90°", "75°", "60°", "45°", "30°", "15°", "0°"], fontsize=7)
    ax.set_rlabel_position(135)

    station_names = [d["station"] for d in all_station_data]
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(station_names), 1)))
    stn_color = {s: palette[i] for i, s in enumerate(station_names)}

    slip_scatter_az = []
    slip_scatter_el = []

    for sdata in all_station_data:
        station = sdata["station"]
        rx_xyz = sdata["rx_xyz"]
        file_start = sdata["file_start"]
        file_end = sdata["file_end"]
        svids = sdata["svids"]
        slip_df = sdata["slip_df"]
        color = stn_color[station]

        obs_sec = (file_end - file_start).total_seconds()
        t_samples = np.arange(0, obs_sec + sample_sec, sample_sec)

        for svid in svids:
            azs, els = [], []
            for t_sec in t_samples:
                epoch = file_start + dt.timedelta(seconds=float(t_sec))
                try:
                    satpos = svpos.getSvPos(epoch, const=['G', 'E'])
                    if svid not in satpos.index:
                        continue
                    sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                    elev = elevation_angle(rx_xyz, sat_xyz)
                    if elev < elev_cut:
                        continue
                    az = azimuth_angle(rx_xyz, sat_xyz)
                    azs.append(np.radians(az))
                    els.append(90 - elev)
                except Exception:
                    continue
            if azs:
                style = "-" if svid.startswith("G") else "--"
                ax.plot(azs, els, color=color, linewidth=0.8,
                        alpha=0.5, linestyle=style)

        if not slip_df.empty:
            for _, row in slip_df.iterrows():
                svid = row["svid"]
                epoch = row["epoch"]
                try:
                    satpos = svpos.getSvPos(epoch, const=['G', 'E'])
                    if svid not in satpos.index:
                        continue
                    sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                    elev = elevation_angle(rx_xyz, sat_xyz)
                    if elev < elev_cut:
                        continue
                    az = azimuth_angle(rx_xyz, sat_xyz)
                    slip_scatter_az.append(np.radians(az))
                    slip_scatter_el.append(90 - elev)
                except Exception:
                    continue

    if slip_scatter_az:
        ax.scatter(slip_scatter_az, slip_scatter_el,
                   color="red", s=18, zorder=8, alpha=0.7)

    legend_handles = [
        Line2D([0], [0], color=stn_color[s], linewidth=1.5, label=s)
        for s in station_names
    ]
    legend_handles += [
        Line2D([0], [0], color="grey", linewidth=1, linestyle="-", label="GPS"),
        Line2D([0], [0], color="grey", linewidth=1, linestyle="--", label="Galileo"),
        Line2D([0], [0], color="red", marker="o", linestyle="None",
               markersize=5, label="Cycle slip"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.45, 1.12), fontsize=7, title="Station / system")
    ax.set_title("Skyplot all stations\nSolid=GPS  Dashed=Galileo  Red=cycle slip",
                 pad=20, fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "ALL_skyplot.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved combined skyplot: {out}")


# ---------------------------------------------------------------------------
# Combined timeline GF only, fixed size
# ---------------------------------------------------------------------------
def make_combined_timeline(pkl_files, out_dir, max_rows=300):
    """
    One row per satellite (union across all stations).
    Red dots = every slip detected on that satellite by any station.
    """
    # collect per-satellite: t_range and slip times (from any station)
    sat_t0 = {}   # svid -> min t0 across stations
    sat_t1 = {}   # svid -> max t1 across stations
    sat_slips = {}   # svid -> list of t_sec values
    sat_const = {}   # svid -> "G" or "E"

    for pkl_path in sorted(pkl_files):
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        pgf = saved["plot_data_gf"]

        for svid, (t_gf, _, slips, _) in pgf.items():
            if len(t_gf) == 0:
                continue
            const = "G" if svid.startswith("G") else "E"
            sat_const[svid] = const

            t0, t1 = t_gf[0], t_gf[-1]
            sat_t0[svid] = min(sat_t0.get(svid, t0), t0)
            sat_t1[svid] = max(sat_t1.get(svid, t1), t1)

            if svid not in sat_slips:
                sat_slips[svid] = []
            if len(slips) > 0:
                sat_slips[svid].extend(t_gf[slips].tolist())

    # order: GPS first, then Galileo, alphabetical within each
    gps_svids = sorted(s for s in sat_const if sat_const[s] == "G")
    gal_svids = sorted(s for s in sat_const if sat_const[s] == "E")
    ordered = gps_svids + gal_svids

    if not ordered:
        print("No entries for combined timeline.")
        return

    n_rows = len(ordered)
    fig_h = min(60, max(6, n_rows * 0.35))
    fig, ax = plt.subplots(figsize=(16, fig_h))

    for yi, svid in enumerate(ordered):
        const = sat_const[svid]
        color = "steelblue" if const == "G" else "darkorange"
        t0, t1 = sat_t0[svid], sat_t1[svid]
        ax.plot([t0, t1], [yi, yi], color=color, linewidth=1.2, alpha=0.5)

        slips = sat_slips.get(svid, [])
        if slips:
            ax.scatter(slips, np.full(len(slips), yi),
                       color="red", s=14, zorder=5, alpha=0.8,
                       marker="o", linewidths=0)

    # separator between GPS and Galileo
    if gps_svids and gal_svids:
        ax.axhline(len(gps_svids) - 0.5, color="grey",
                   linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ordered, fontsize=7)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Time from file start (s)", fontsize=10)
    ax.set_title(
        "Cycle Slip Timeline all stations combined, one row per satellite\n"
        "Blue = GPS   Orange = Galileo   Red dot = slip (any station)",
        fontsize=11,
    )

    ax.legend(handles=[
        Line2D([0], [0], color="steelblue",  linewidth=1.5, label="GPS"),
        Line2D([0], [0], color="darkorange", linewidth=1.5, label="Galileo"),
        Line2D([0], [0], color="red", marker="o", linestyle="None",
               markersize=5, label="Cycle slip (any station)"),
    ], fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(out_dir, "ALL_timeline_combined.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved combined timeline: {out}  ({n_rows} satellites)")


# ===========================================================================
# MAIN LOOP
# ===========================================================================
lat_edges, lon_edges = build_grid()
pkl_files = sorted(
    p for p in glob.glob(os.path.join("results", "*_slip_results.pkl"))
    if os.path.basename(p) != "all_slip_results.pkl"
)
print(f"Found {len(pkl_files)} station result files in 'results/'")

all_slip_points = []
all_ipp_points = []
all_station_data = []

for pkl_path in pkl_files:

    station = os.path.basename(pkl_path).replace("_slip_results.pkl", "")
    print(f"\n{'='*50}\nMapping station: {station}\n{'='*50}")

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    plot_data_gf = saved["plot_data_gf"]
    file_start = saved["file_start"]
    file_end = saved.get("file_end", file_start + dt.timedelta(hours=24))

    rinex_path = os.path.join("data", f"{station.lower()}1320.24o")
    rx_xyz = get_receiver_xyz(rinex_path)
    if rx_xyz is None:
        print(f"  Could not find RINEX for {station}, skipping")
        continue

    obs_sec = (file_end - file_start).total_seconds()
    obs_hours = obs_sec / 3600.0
    svids = list(plot_data_gf.keys())

    ipp_df = build_ipp_obs(svids, file_start, file_end, rx_xyz)

    slip_points = []
    for svid, (t_gf, res_gf, slips, lam1_sv) in plot_data_gf.items():
        if len(slips) == 0:
            continue
        for slip_idx in slips:
            t_sec = float(t_gf[slip_idx])
            epoch = file_start + dt.timedelta(seconds=t_sec)
            tw_idx = int(t_sec // TIME_WINDOW_SEC)
            try:
                satpos = svpos.getSvPos(epoch, const=['G', 'E'])
                if svid not in satpos.index:
                    continue
                sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                elev = elevation_angle(rx_xyz, sat_xyz)
                if elev < 15:
                    continue
                ipp = ionospheric_pierce_point(rx_xyz, sat_xyz)
                if ipp is None:
                    continue
                lat, lon = ecef_to_latlon(ipp)
                slip_points.append({
                    "station": station, "svid": svid,
                    "epoch": epoch, "time_window": tw_idx,
                    "lat": lat, "lon": lon, "n_slips": 1,
                })
            except Exception as e:
                print(f"  {svid} at {epoch}: {e}")

    slip_df = pd.DataFrame(slip_points)
    print(f"Total mapped slips: {len(slip_df)}")

    all_slip_points.extend(slip_points)
    all_ipp_points.extend(ipp_df.to_dict("records"))
    all_station_data.append({
        "station": station, "svids": svids,
        "file_start": file_start, "file_end": file_end,
        "rx_xyz": rx_xyz, "slip_df": slip_df,
    })

    if slip_df.empty:
        print(f"  No slips to plot for {station}, skipping maps")
        continue

    grid_count = bin_slips(slip_df, "n_slips", lat_edges, lon_edges, "sum")

    # Map: raw counts
    fig = plt.figure(figsize=(8, 8))
    ax = make_polar_ax(fig)
    vmax = np.nanmax(grid_count) if not np.all(np.isnan(grid_count)) else 1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    pcm  = plot_binned_grid(ax, grid_count, lon_edges, lat_edges,
                            cmap="Reds", norm=norm)
    plt.colorbar(pcm, ax=ax, label="Total cycle slips", shrink=0.7, pad=0.06)
    ax.set_title(f"Cycle-Slip Count at IPP: {station}\n", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{station}_map_count.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved maps for {station}")


    # Map: scatter (one dot per slip IPP)
    fig = plt.figure(figsize=(8, 8))
    ax = make_polar_ax(fig)
    ax.scatter(slip_df["lon"].values, slip_df["lat"].values,
               s=12, c="red", alpha=0.6, transform=DATA_CRS, zorder=5)
    ax.set_title(f"Cycle-Slip IPP positions: {station}\n"
                 f"one dot = one detected slip", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{station}_map_scatter.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved maps for {station}")


    # Map skyplot for each station
    make_single_station_skyplot(station, rx_xyz, svids,
                                file_start, file_end, plot_data_gf,
                                svpos, OUT_DIR)



# ===========================================================================
# COMBINED all stations
# ===========================================================================
all_slip_df = pd.DataFrame(all_slip_points)
all_ipp_df = pd.DataFrame(all_ipp_points)
print(f"\nTotal slips: {len(all_slip_df)}   Total IPP obs: {len(all_ipp_df)}")

if not all_slip_df.empty:

    grid_all = bin_slips(all_slip_df, "n_slips", lat_edges, lon_edges, "sum")

    # --- combined total count map ---
    fig = plt.figure(figsize=(8, 8))
    ax  = make_polar_ax(fig)
    vmax = np.nanmax(grid_all) if not np.all(np.isnan(grid_all)) else 1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    pcm  = plot_binned_grid(ax, grid_all, lon_edges, lat_edges,
                            cmap="Reds", norm=norm)
    plt.colorbar(pcm, ax=ax, label="Total cycle slips", shrink=0.7, pad=0.06)
    ax.set_title(f"Cycle-Slip Count all stations\n", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ALL_map_count.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # --- combined scatter ---
    fig = plt.figure(figsize=(8, 8))
    ax = make_polar_ax(fig)
    ax.scatter(all_slip_df["lon"].values, all_slip_df["lat"].values,
               s=10, c="red", alpha=0.5, transform=DATA_CRS, zorder=5)
    ax.set_title("Cycle-Slip IPP positions: all stations\n"
                 "one dot = one detected slip", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ALL_map_scatter.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # --- combined per-window: only windows that have at least one slip ---
    active_windows = sorted(all_slip_df["time_window"].unique())  # <-- only non-empty
    n_active = len(active_windows)
    print(f"Time windows with slips: {n_active}  "
          f"(indices: {active_windows})")

    if n_active > 0:
        tw_grids  = []
        tw_labels = []
        for tw in active_windows:
            df_tw = all_slip_df[all_slip_df["time_window"] == tw]
            g = bin_slips(df_tw, "n_slips", lat_edges, lon_edges, "sum")
            tw_grids.append(g)
            t0h = tw * TIME_WINDOW_SEC / 3600
            t1h = t0h + TIME_WINDOW_SEC / 3600
            tw_labels.append(f"{t0h:.0f}–{t1h:.0f} h UTC")

        valid_maxes = [np.nanmax(r) for r in tw_grids
                       if not np.all(np.isnan(r))]
        tw_vmax = max(valid_maxes) if valid_maxes else 1
        norm_tw = mcolors.Normalize(vmin=0, vmax=tw_vmax)
        ncols   = min(4, n_active)
        nrows   = int(np.ceil(n_active / ncols))

        fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
        for i, (g, label) in enumerate(zip(tw_grids, tw_labels)):
            ax = add_polar_ax_to_figure(fig, nrows, ncols, i + 1)
            plot_binned_grid(ax, g, lon_edges, lat_edges,
                             cmap="Reds", norm=norm_tw)
            ax.set_title(label, fontsize=8)

        fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm_tw)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Cycle-slip count")
        fig.suptitle(
            f"Cycle-Slip Count per 1-h Window all stations\n"
            f"({n_active} windows with detected slips, "
            f"empty windows omitted)",
            fontsize=11,
        )
        plt.savefig(os.path.join(OUT_DIR, "ALL_map_windows.png"),
                    dpi=110, bbox_inches="tight")
        plt.close()

    print("Saved ALL_map_count / ALL_map_windows / ALL_map_scatter")

# combined skyplot
make_combined_skyplot(all_station_data, svpos, OUT_DIR)

# combined timeline
make_combined_timeline(pkl_files, OUT_DIR)

