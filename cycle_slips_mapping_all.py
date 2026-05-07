import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
TIME_WINDOW_SEC = 3600 #1-hour bins for per-panel maps

GRID_DLAT = 2.0 #spatial grid resolution [degrees]
GRID_DLON = 5.0

H_IONO  = 350000.0 #ionospheric shell height [m]
R_EARTH = 6371000.0

# Cadence for n_obs sampling [seconds].
# Set equal to the SP3 file cadence (300 s = 5 min) so we call getSvPos
# only once per SP3 epoch per satellite
OBS_SAMPLE_SEC = 300

# ---------------------------------------------------------------------------
# SP3 orbits
# ---------------------------------------------------------------------------
svpos = so.sp3Orbits("data/COD0MGXFIN_20241320000_01D_05M_ORB.SP3")

# ---------------------------------------------------------------------------
# Geometry functions
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
    t_candidates = [t for t in [(-b + np.sqrt(disc)) / 2,
                                 (-b - np.sqrt(disc)) / 2] if t > 0]
    if not t_candidates:
        return None
    return rx_xyz + min(t_candidates) * u


def elevation_angle(rx_xyz, sat_xyz):
    rx_up = rx_xyz / np.linalg.norm(rx_xyz)
    los   = sat_xyz - rx_xyz
    los   = los / np.linalg.norm(los)
    return np.degrees(np.arcsin(np.dot(rx_up, los)))


def mapping_function(elev_deg, h_iono=H_IONO):
    zeta = np.radians(90.0 - elev_deg)
    return 1.0 / np.sqrt(
        1.0 - ((R_EARTH * np.sin(zeta)) / (R_EARTH + h_iono)) ** 2
    )


# ---------------------------------------------------------------------------
# Map functions
# ---------------------------------------------------------------------------
POLAR_PROJ = ccrs.NorthPolarStereo(central_longitude=-40.0)
DATA_CRS   = ccrs.PlateCarree()


def _add_features(ax, full=False):
    """
    full=False -> zoomed on Greenland [-80, 10, 58, 85]
    full=True -> full Arctic view [-180, 180, 50, 90]
    """
    if full:
        ax.set_extent([-180, 180, 50, 90], crs=DATA_CRS)
    else:
        ax.set_extent([-80, 10, 58, 85], crs=DATA_CRS)
    ax.add_feature(cfeature.LAND, alpha=0.4)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="grey", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return ax


def make_polar_ax(fig, position=111, full=False):
    ax = fig.add_subplot(position, projection=POLAR_PROJ)
    return _add_features(ax, full=full)


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


# ---------------------------------------------------------------------------
# Spatial bin functions
# ---------------------------------------------------------------------------

def build_grid(lat_min=50, lat_max=90, lon_min=-180, lon_max=180, dlat=GRID_DLAT, dlon=GRID_DLON):
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

    mask = ((lat_idx >= 0) & (lat_idx < n_lat) & (lon_idx >= 0) & (lon_idx < n_lon))
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
    return ax.pcolormesh(
        lon_edges, lat_edges, grid,
        cmap=cmap, norm=norm,
        transform=DATA_CRS,
        shading="flat",
    )


# ---------------------------------------------------------------------------
# n_obs function
# ---------------------------------------------------------------------------

def build_ipp_obs(svids, file_start, file_end, rx_xyz,
                       sample_sec=OBS_SAMPLE_SEC):
    """
    Sample satellite positions on a time grid (5 min) and compute IPP 
    lat/lon for every visible satellite at each sample epoch. 
    Returns a DataFrame with columns lat, lon, time_window,
    n_obs. one row per (satellite, sample epoch) that passes the elevation
    cut
    """
    obs_sec = (file_end - file_start).total_seconds()
    t_samples = np.arange(0, obs_sec + sample_sec, sample_sec)

    rows = []
    for t_sec in t_samples:
        epoch = file_start + dt.timedelta(seconds=float(t_sec))
        tw_idx = int(t_sec // TIME_WINDOW_SEC)

        try:
            satpos = svpos.getSvPos(epoch)
        except Exception:
            continue

        for svid in svids:
            if svid not in satpos.index:
                continue
            try:
                sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                elev = elevation_angle(rx_xyz, sat_xyz)
                if elev < 15:
                    continue
                ipp = ionospheric_pierce_point(rx_xyz, sat_xyz)
                if ipp is None:
                    continue
                lat, lon = ecef_to_latlon(ipp)
                rows.append({
                    "lat": lat,
                    "lon": lon,
                    "time_window": tw_idx,
                    "n_obs": 1,
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


# ------------------
# Output directory
# -----------------
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

lat_edges, lon_edges = build_grid()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
pkl_files = sorted(glob.glob("results/*_slip_results.pkl"))
print(f"Found {len(pkl_files)} station result files in 'results/'")

all_slip_points = []
all_ipp_points  = []

for pkl_path in pkl_files:

    station = os.path.basename(pkl_path).replace("_slip_results.pkl", "")
    print(f"\n{'='*50}\nMapping station: {station} ({pkl_path})\n{'='*50}")

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    plot_data_gf = saved["plot_data_gf"]
    file_start = saved["file_start"]
    file_end = saved.get("file_end", file_start + dt.timedelta(hours=24))

    rinex_path = os.path.join("data", f"{station.lower()}1320.24o")
    rx_xyz = get_receiver_xyz(rinex_path)
    if rx_xyz is None:
        print(f"  Could not find RINEX file for {station}, skipping")
        continue
    print(f"Receiver XYZ: {rx_xyz}")

    obs_duration_sec = (file_end - file_start).total_seconds()
    obs_hours = obs_duration_sec / 3600.0
    n_windows = max(1, int(np.ceil(obs_duration_sec / TIME_WINDOW_SEC)))
    print(f"Observation span: {obs_hours:.1f} h  ->  "
          f"{n_windows} windows of {TIME_WINDOW_SEC/3600:.1f} h")

    # -----------------------------------------------------------------------
    # 1: n_obs on coarse grid
    # -----------------------------------------------------------------------
    svids = list(plot_data_gf.keys())
    n_sample_epochs = int(obs_duration_sec / OBS_SAMPLE_SEC) + 1
    print(f"Building n_obs: {len(svids)} SVs x {n_sample_epochs} epochs "
          f"({OBS_SAMPLE_SEC}s cadence)  [was {len(svids)*int(obs_duration_sec/30)} calls before]")

    ipp_df = build_ipp_obs(svids, file_start, file_end, rx_xyz)
    print(f"Total IPP passages sampled: {len(ipp_df)}")

    # -----------------------------------------------------------------------
    # 2: slip IPP positions
    # -----------------------------------------------------------------------
    slip_points = []

    for svid, (t_gf, res_gf, slips, lam1_sv) in plot_data_gf.items():
        if len(slips) == 0:
            continue

        for slip_idx in slips:
            t_sec = float(t_gf[slip_idx])
            epoch = file_start + dt.timedelta(seconds=t_sec)
            tw_idx = int(t_sec // TIME_WINDOW_SEC)

            try:
                satpos = svpos.getSvPos(epoch)
                if svid not in satpos.index:
                    continue
                sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)
                elev = elevation_angle(rx_xyz, sat_xyz)
                if elev < 15:
                    continue
                mf = mapping_function(elev)
                ipp = ionospheric_pierce_point(rx_xyz, sat_xyz)
                if ipp is None:
                    continue
                lat, lon = ecef_to_latlon(ipp)
                slip_points.append({
                    "station": station,
                    "svid": svid,
                    "epoch": epoch,
                    "time_window": tw_idx,
                    "lat": lat,
                    "lon": lon,
                    "residual": res_gf[slip_idx],
                    "elevation": elev,
                    "mapping_function": mf,
                    "vertical_residual": res_gf[slip_idx] / mf,
                    "n_slips":1,
                })
            except Exception as e:
                print(f"Could not process {svid} at {epoch}: {e}")

    slip_df = pd.DataFrame(slip_points)
    print(f"Total mapped slips: {len(slip_df)}")

    all_slip_points.extend(slip_points)
    all_ipp_points.extend(ipp_df.to_dict("records"))

    if slip_df.empty:
        print(f" No slips to plot for {station}, skipping")
        continue

    # -----------------------------------------------------------------------
    # Build grids
    # -----------------------------------------------------------------------
    grid_count = bin_slips(slip_df, "n_slips",
                           lat_edges, lon_edges, aggregator="sum")
    grid_vert = bin_slips(slip_df, "vertical_residual",
                           lat_edges, lon_edges, aggregator="mean")
    grid_obs = bin_slips(ipp_df,  "n_obs",
                           lat_edges, lon_edges, aggregator="sum")

    with np.errstate(invalid="ignore"):
        grid_rate_hour = np.where(
            ~np.isnan(grid_count), grid_count / obs_hours, np.nan
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        grid_rate_nobs = np.where(
            (~np.isnan(grid_count)) & (~np.isnan(grid_obs)) & (grid_obs > 0),
            grid_count / grid_obs,
            np.nan,
        )

    # -----------------------------------------------------------------------
    # Map 1: slip/hour
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = make_polar_ax(fig)
    vmax = np.nanmax(grid_rate_hour) if not np.all(np.isnan(grid_rate_hour)) else 1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    pcm  = plot_binned_grid(ax, grid_rate_hour, lon_edges, lat_edges,
                            cmap="Reds", norm=norm)
    plt.colorbar(pcm, ax=ax, label="Cycle-slip rate [slips/hour]",
                 shrink=0.7, pad=0.06)
    ax.set_title(
        f"Cycle-Slip Rate at IPP: {station}\n"
        f"Grid: {GRID_DLAT}°x{GRID_DLON}°  |  "
        f"Aggregated over {obs_hours:.1f} h total observation",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{station}_map_rate_hour.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Map 2: slip/n_obs
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = make_polar_ax(fig)
    vmax2 = np.nanmax(grid_rate_nobs) if not np.all(np.isnan(grid_rate_nobs)) else 1
    norm2 = mcolors.Normalize(vmin=0, vmax=vmax2)
    pcm2 = plot_binned_grid(ax, grid_rate_nobs, lon_edges, lat_edges,
                             cmap="Reds", norm=norm2)
    plt.colorbar(pcm2, ax=ax,
                 label="Cycle-slip rate [slips / satellite passage]",
                 shrink=0.7, pad=0.06)
    ax.set_title(
        f"Cycle-Slip Rate normalised by Observations at IPP: {station}\n"
        f"Grid: {GRID_DLAT}°x{GRID_DLON}°  |  slips / satellite passage per cell",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{station}_map_rate_nobs.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Map 3: mean vertical GF residual
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = make_polar_ax(fig)
    vmax3 = np.nanmax(np.abs(grid_vert)) if not np.all(np.isnan(grid_vert)) else 1
    norm3 = mcolors.Normalize(vmin=0, vmax=vmax3)
    pcm3 = plot_binned_grid(ax, np.abs(grid_vert), lon_edges, lat_edges,
                             cmap="viridis", norm=norm3)
    plt.colorbar(pcm3, ax=ax, label="|Mean vertical GF residual| [m]",
                 shrink=0.7, pad=0.06)
    ax.set_title(
        f"Ionospheric Disturbance Intensity at IPP: {station}\n"
        f"Grid: {GRID_DLAT}°x{GRID_DLON}°  |  mean vertical GF residual",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{station}_map_intensity.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Map 4: per-window panels (slip/hour)
    # -----------------------------------------------------------------------
    tw_window_hours = TIME_WINDOW_SEC / 3600.0

    per_window_grids = []
    for tw in range(n_windows):
        df_tw = slip_df[slip_df["time_window"] == tw]
        g = bin_slips(df_tw, "n_slips", lat_edges, lon_edges, "sum")
        rate = np.where(~np.isnan(g), g / tw_window_hours, np.nan)
        per_window_grids.append(rate)

    valid_maxes = [np.nanmax(r) for r in per_window_grids
                   if not np.all(np.isnan(r))]
    tw_vmax = max(valid_maxes) if valid_maxes else 1
    norm_tw = mcolors.Normalize(vmin=0, vmax=tw_vmax)

    ncols = min(4, n_windows)
    nrows = int(np.ceil(n_windows / ncols))
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for tw, g_rate in enumerate(per_window_grids):
        t_start_h = tw * tw_window_hours
        t_end_h = t_start_h + tw_window_hours
        ax = add_polar_ax_to_figure(fig, nrows, ncols, tw + 1)
        plot_binned_grid(ax, g_rate, lon_edges, lat_edges,
                         cmap="Reds", norm=norm_tw)
        ax.set_title(f"Win {tw+1}: {t_start_h:.0f}-{t_end_h:.0f} h UTC",
                     fontsize=8)

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm_tw)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Cycle-slip rate [slips/hour]")
    fig.suptitle(
        f"Cycle-Slip Rate per {TIME_WINDOW_SEC//3600}-h Window: {station}\n"
        "(each panel = 1 h, IPP positions evolve as satellites move)",
        fontsize=11,
    )
    plt.savefig(os.path.join(OUT_DIR, f"{station}_map_windows.png"),
                dpi=110, bbox_inches="tight")
    plt.close()

    print(f"Saved all maps for {station}")


# ---------------------------------------------------------------------------
# Combined maps, all stations
# ---------------------------------------------------------------------------
all_slip_df = pd.DataFrame(all_slip_points)
all_ipp_df = pd.DataFrame(all_ipp_points)
print(f"\nTotal slips across all stations: {len(all_slip_df)}")
print(f"Total IPP passages across all stations: {len(all_ipp_df)}")

if not all_slip_df.empty:

    grid_all_count = bin_slips(all_slip_df, "n_slips",
                               lat_edges, lon_edges, aggregator="sum")
    grid_all_vert = bin_slips(all_slip_df, "vertical_residual",
                               lat_edges, lon_edges, aggregator="mean")
    grid_all_obs = bin_slips(all_ipp_df,  "n_obs",
                               lat_edges, lon_edges, aggregator="sum")

    obs_hours_all = 24.0

    with np.errstate(invalid="ignore"):
        grid_all_rate_hour = np.where(
            ~np.isnan(grid_all_count),
            grid_all_count / obs_hours_all, np.nan,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        grid_all_rate_nobs = np.where(
            (~np.isnan(grid_all_count)) &
            (~np.isnan(grid_all_obs))   &
            (grid_all_obs > 0),
            grid_all_count / grid_all_obs, np.nan,
        )

    for fname, grid, cmap, label, title in [
        (
            "ALL_map_rate_hour.png",
            grid_all_rate_hour, "Reds",
            "Cycle-slip rate [slips/hour]",
            f"Cycle-Slip Rate at IPP: all stations\n"
            f"Grid: {GRID_DLAT}°x{GRID_DLON}°  |  Aggregated over {obs_hours_all:.0f} h",
        ),
        (
            "ALL_map_rate_nobs.png",
            grid_all_rate_nobs, "Reds",
            "Cycle-slip rate [slips / satellite passage]",
            f"Cycle-Slip Rate normalised by Observations: all stations\n"
            f"Grid: {GRID_DLAT}°x{GRID_DLON}°  |  slips / satellite passage per cell",
        ),
        (
            "ALL_map_intensity.png",
            np.abs(grid_all_vert), "viridis",
            "|Mean vertical GF residual| [m]",
            f"Ionospheric Disturbance Intensity at IPP — all stations\n"
            f"Grid: {GRID_DLAT}°x{GRID_DLON}°  |  mean vertical GF residual",
        ),
    ]:
        fig = plt.figure(figsize=(8, 8))
        ax = make_polar_ax(fig)
        vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        pcm = plot_binned_grid(ax, grid, lon_edges, lat_edges, cmap=cmap, norm=norm)
        plt.colorbar(pcm, ax=ax, label=label, shrink=0.7, pad=0.06)
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches="tight")
        plt.close()

    # per-window combined
    all_windows = sorted(all_slip_df["time_window"].unique())
    n_tw_all = len(all_windows)

    if n_tw_all > 0:
        tw_window_hours = TIME_WINDOW_SEC / 3600.0
        tw_rates_all = []
        for tw in all_windows:
            g = bin_slips(
                all_slip_df[all_slip_df["time_window"] == tw],
                "n_slips", lat_edges, lon_edges, "sum"
            )
            tw_rates_all.append(
                np.where(~np.isnan(g), g / tw_window_hours, np.nan)
            )

        valid_maxes = [np.nanmax(r) for r in tw_rates_all
                       if not np.all(np.isnan(r))]
        tw_vmax = max(valid_maxes) if valid_maxes else 1
        norm_tw = mcolors.Normalize(vmin=0, vmax=tw_vmax)

        ncols = min(4, n_tw_all)
        nrows = int(np.ceil(n_tw_all / ncols))
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

        for i, (tw_idx, g_rate) in enumerate(zip(all_windows, tw_rates_all)):
            t_start_h = tw_idx * tw_window_hours
            t_end_h = t_start_h + tw_window_hours
            ax = add_polar_ax_to_figure(fig, nrows, ncols, i + 1)
            plot_binned_grid(ax, g_rate, lon_edges, lat_edges,
                             cmap="Reds", norm=norm_tw)
            ax.set_title(
                f"Win {tw_idx+1}: {t_start_h:.0f}-{t_end_h:.0f} h UTC",
                fontsize=8,
            )

        fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm_tw)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Cycle-slip rate [slips/hour]")
        fig.suptitle(
            f"Cycle-Slip Rate per {TIME_WINDOW_SEC//3600}-h Window — all stations\n"
            "(IPP positions evolve as satellites move)",
            fontsize=11,
        )
        plt.savefig(os.path.join(OUT_DIR, "ALL_map_windows.png"),
                    dpi=110, bbox_inches="tight")
        plt.close()

    print("\nSaved all combined maps to results/ALL_map_*.png")
    print(
        f"\nMetrics used:\n"
        f"  slip/hour   = slips in cell / {obs_hours_all:.0f} h\n"
        f"  slip/n_obs  = slips in cell / satellite passages in cell "
        f"(sampled every {OBS_SAMPLE_SEC}s)\n"
        f"  GF residual = mean |vertical GF residual| per cell [m]\n"
        f"  Time windows: {TIME_WINDOW_SEC//3600}-h bins "
        f"({n_tw_all} windows in combined dataset)"
    )