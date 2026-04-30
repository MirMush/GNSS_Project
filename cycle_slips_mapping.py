import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SatOrbits as so
import datetime as dt
import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# Load detection results saved by cycle_slips_detection.py
# ---------------------------------------------------------------------------
with open("data/slip_results.pkl", "rb") as f:
    saved = pickle.load(f)

plot_data_gf = saved["plot_data_gf"]
file_start   = saved["file_start"]

# ---------------------------------------------------------------------------
# Receiver position (from RINEX header)
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

rx_xyz = get_receiver_xyz("data/nuuk1320.24o")
print(f"Receiver XYZ: {rx_xyz}")

# ---------------------------------------------------------------------------
# SP3 orbits
# ---------------------------------------------------------------------------
svpos = so.sp3Orbits("data/COD0MGXFIN_20241320000_01D_05M_ORB.SP3")

epoch = dt.datetime(2024, 5, 11, 0, 0, 0)
satpos  = svpos.getSvPos(epoch)
sat_xyz = satpos.loc["G05", ["X", "Y", "Z"]].values
rho     = np.linalg.norm(sat_xyz - rx_xyz)
print(f"G05 position: {sat_xyz}")
print(f"Distance receiver-G05: {rho/1000:.1f} km")

# ---------------------------------------------------------------------------
# Ionospheric Pierce Point (IPP) at 350 km
# ---------------------------------------------------------------------------
R_EARTH = 6371000.0
H_IONO  = 350000.0


def ecef_to_latlon(xyz):
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    return np.degrees(np.arcsin(z / r)), np.degrees(np.arctan2(y, x))


def ionospheric_pierce_point(rx_xyz, sat_xyz, h_iono=350000.0):
    R = R_EARTH + h_iono
    u = sat_xyz - rx_xyz
    u = u / np.linalg.norm(u)

    b = 2 * np.dot(rx_xyz, u)
    c = np.dot(rx_xyz, rx_xyz) - R**2
    disc = b**2 - 4 * c
    if disc < 0:
        return None

    t_candidates = [t for t in [(-b + np.sqrt(disc)) / 2,
                                 (-b - np.sqrt(disc)) / 2] if t > 0]
    if not t_candidates:
        return None

    return rx_xyz + min(t_candidates) * u


def elevation_angle(rx_xyz, sat_xyz):
    rx_up = rx_xyz / np.linalg.norm(rx_xyz)

    los = sat_xyz - rx_xyz
    los = los / np.linalg.norm(los)

    elev_rad = np.arcsin(np.dot(rx_up, los))
    return np.degrees(elev_rad)


def mapping_function(elev_deg, h_iono=350000.0):
    R_E = 6371000.0

    zeta = np.radians(90.0 - elev_deg)

    mf = 1.0 / np.sqrt(
        1.0 - ((R_E * np.sin(zeta)) / (R_E + h_iono))**2
    )

    return mf


# ---------------------------------------------------------------------------
# Map slip locations to IPPs
# ---------------------------------------------------------------------------
slip_points = []

for svid, (t_gf, res_gf, slips, lam1_sv) in plot_data_gf.items():
    if len(slips) == 0:
        continue

    for slip_idx in slips:
        epoch = file_start + dt.timedelta(seconds=float(t_gf[slip_idx]))

        try:
            satpos = svpos.getSvPos(epoch)
            if svid not in satpos.index:
                continue

            sat_xyz = satpos.loc[svid, ["X", "Y", "Z"]].values.astype(float)

            elev = elevation_angle(rx_xyz, sat_xyz)

            # evitar satélites muito baixos
            if elev < 15:
                continue

            mf = mapping_function(elev, H_IONO)

            vertical_residual = res_gf[slip_idx] / mf
            ipp_xyz = ionospheric_pierce_point(rx_xyz, sat_xyz, H_IONO)
            if ipp_xyz is None:
                continue

            lat, lon = ecef_to_latlon(ipp_xyz)
            slip_points.append({
                "svid":             svid,
                "epoch":            epoch,
                "lat":              lat,
                "lon":              lon,
                "residual":         res_gf[slip_idx],
                "elevation":        elev,
                "mapping_function": mf,
                "vertical_residual": res_gf[slip_idx] / mf,
            })

        except Exception as e:
            print(f"Could not process {svid} at {epoch}: {e}")

slip_df = pd.DataFrame(slip_points)
print(slip_df.head())
print(f"Total mapped slips: {len(slip_df)}")



# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Map of slip locations 

fig = plt.figure(figsize=(9, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, -10, 50, 85], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,      alpha=0.4)
ax.add_feature(cfeature.OCEAN,     alpha=0.3)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS,   linestyle=":")

if not slip_df.empty:
    sc = ax.scatter(
        slip_df["lon"], slip_df["lat"],
        c=np.abs(slip_df["residual"]),
        s=35, cmap="Reds", edgecolor="black",
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(sc, ax=ax, label="|GF residual| [m]")

    for _, row in slip_df.iterrows():
        ax.text(row["lon"], row["lat"], row["svid"],
                fontsize=7, transform=ccrs.PlateCarree())

ax.set_title("Cycle Slips mapped at ionospheric height 350 km")
ax.gridlines(draw_labels=True)
plt.show()



# Map of intesity of ionospheric disturbances (GF residuals) at IPPs

fig = plt.figure(figsize=(9, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-90, -10, 50, 85], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, alpha=0.4)
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

if not slip_df.empty:
    sc = ax.scatter(
        slip_df["lon"],
        slip_df["lat"],
        c=np.abs(slip_df["vertical_residual"]),
        s=35,
        cmap="viridis",
        edgecolor="black",
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc, ax=ax, label="|Vertical GF residual| [m]")

ax.set_title("Cycle Slip Intensity corrected by Mapping Function")
ax.gridlines(draw_labels=True)

plt.show()