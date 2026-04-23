import hatanaka
from rinexreader2 import rinexReader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load file
filepath = "nuuk1320.24d"

#decompress file in rinex2
rnx_2 = hatanaka.decompress_on_disk(filepath)

#read file
rnx = rinexReader("nuuk1320.24o")
rnx.readFile(
    readConst=["G"],
    sigTypes=["L1", "L2", "C1", "P2"]
)

print("Start:", rnx.fileStart)
print("End:", rnx.fileEnd)
print("Epochs:", len(rnx.obs))
print("Satellites:", len(rnx.obsSvid))

g05 = rnx.get_svid_data("G05", ["L1", "L2"])
print(g05.head())

# function to calculate moving average
def movmedian(data, window):
    return pd.Series(data).rolling(window=int(window), center=True, min_periods=1).median().values

def identify_cycle_slips(tow, phase):

    # compute phase increments
    dphi = np.diff(phase)
    dphi = np.insert(dphi, 0, 0)
    if len(dphi) >= 2:
        dphi[0] = dphi[1]

    # normalize by dt
    dt = np.diff(tow)
    dt = np.insert(dt, 0, dt[0])
    dphi = dphi / dt

    # remove repeated or invalid samples
    repeated_or_nan = (dphi==0) | np.isnan(dphi) | np.isnan(tow) | np.isnan(phase)
    keep_idx = ~repeated_or_nan
    t_clean = tow[keep_idx]
    dphi_clean = dphi[keep_idx]

    # we don't need 4.4 I think

    # construct an adaptive moving-median reference
    medwidth = 60
    mphi = movmedian(dphi_clean, medwidth)
    
    # calculate residuals
    residual = dphi_clean - mphi

    # identify candidate cycle slips
    # slip_thresh = np.pi/2
    slip_thresh = 1.0
    ind_slips = np.where(np.abs(dphi_clean - mphi) > slip_thresh)[0]

    return ind_slips, t_clean, dphi_clean, mphi, residual


# for all the satellites
results = {}
for svid in rnx.obsSvid:
    print(f"Processing satellite {svid}: ")

    # current satellite
    sat_data = rnx.get_svid_data(svid, ["L1"])

    if sat_data.empty or 'L1' not in sat_data.columns:
        print(f"No L1 data for {svid}")
        continue

    tow = (sat_data.index - sat_data.index[0]).total_seconds().values
    phase_l1 = sat_data['L1'].values

    # identify cycle slips
    slips, t_c, d_c, m_c, res = identify_cycle_slips(tow, phase_l1)

    # save results
    results[svid] = len(slips)
    print(f"Found {len(slips)} cycle slips for {svid}")

    # Plot (a lot, maybe comment this)
    if len(slips) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(t_c, res, color='black', label=f'Residuals {svid}')
        plt.scatter(t_c[slips], res[slips], color='red', label='Slips found')
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.axhline(y=-1.0, color='r', linestyle='--')
        plt.title(f"Cycle Slips: satellite {svid}")
        plt.xlabel("Tow (s)")
        plt.ylabel("Cycles")
        plt.legend()
        plt.ylim([-5, 5])
        plt.show()

# recap
print("\n" + "="*30)
print("Cycle slips found")
print("="*30)
for sv, count in results.items():
    print(f"{sv}: {count} slips")


# example for G05
tow = (g05.index - g05.index[0]).total_seconds().values
phase_l1 = g05['L1'].values
slips_5, t_clean, dphi, mphi, residual = identify_cycle_slips(tow, phase_l1)

print(f"Found {len(slips_5)} cycle slips for G05 (L1)")
if len(slips_5) > 0:
    print(f"Jumps happen at seconds (tow): {t_clean[slips_5]}")


# plot
plt.figure(figsize=(12, 10))
plt.scatter(t_clean, residual, label='Residual (dphi - mphi)', color='black')
plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold (1.0 cycle)')
plt.axhline(y=-1.0, color='r', linestyle='--')
plt.scatter(t_clean[slips_5], residual[slips_5], color='red', zorder=5, label='Detected Slips')
plt.title('Residuals and Detection Threshold')
plt.xlabel('Time (seconds)')
plt.ylabel('Cycles')
plt.ylim([-5, 5])
plt.legend()

plt.tight_layout()
plt.show()