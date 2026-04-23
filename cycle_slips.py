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

# function to detect cycle slips
def identify_cycle_slips(tow, phase):
    '''
    
    '''

    # convert carrier phase to radians
    phi_rad = phase * 2*np.pi

    # compute phase increments
    dphi = np.diff(phi_rad)
    dphi = np.insert(dphi, 0, 0)
    if len(dphi) >= 2:
        dphi[0] = dphi[1]

    # remove repeated or invalid samples
    repeated_or_nan = (dphi==0) | np.isnan(dphi) | np.isnan(tow) | np.isnan(phi_rad)
    keep_idx = ~repeated_or_nan
    t_clean = tow[keep_idx]
    dphi_clean = dphi[keep_idx]

    # we don't need 4.4 I think

    # construct an adaptive moving-median reference
    medfactor = 1.5
    medwidth = 10
    medwith_max = len(dphi_clean) // medfactor
    smoothness_thresh = np.pi/2
    mphi_last = movmedian(dphi_clean, medwidth/2)

    while True:
        medwidth = round(medwidth*medfactor)
        if medwidth > medwith_max:
            break
        mphi = movmedian(dphi_clean, medwidth)
        smoothness = np.max(np.abs(mphi - mphi_last))
        mphi_last = mphi
        if smoothness <= smoothness_thresh:
            break
    mphi = movmedian(dphi_clean, medwidth)
    
    # identify candidate cycle slips
    slip_thresh = np.pi/2
    ind_slips = np.where(np.abs(dphi_clean - mphi) > slip_thresh)[0]

    return ind_slips, t_clean, dphi_clean


# example
tow = (g05.index - g05.index[0]).total_seconds().values
phase_l1 = g05['L1'].values
slips, time_slips, dphi = identify_cycle_slips(tow, phase_l1)

print(f"Found {len(slips)} cycle slips for G05 (L1)")
if len(slips) > 0:
    print(f"Jumps happen at seconds (tow): {time_slips[slips]}")


# plot
plt.figure(figsize=(12,6))
plt.plot(time_slips, dphi, label='Phase increments delta_phi', alpha=0.5)
plt.axhline(y=np.pi/2, color='r', linestyle='--', label='threshold')
plt.axhline(y=-np.pi/2, color='r', linestyle='--')
plt.title('Cycle Slips : satellite G05')
plt.xlabel('Time (seconds)')
plt.ylabel('Radians')
plt.legend()
plt.show()
