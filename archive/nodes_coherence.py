import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import coherence
from obspy import Stream, read, read_inventory
from obspy.signal import PPSD

from helper_funcs import extract_psd_time_series


FS_CPD = 48.0          # samples per day (30-min cadence -> 48 samples/day)
NPERSEG = 96           # Welch segment length = 2 days -> good compromise for 9–10 day record


station = '03479'
comp = 'Z'
st = Stream()
for d in range(1,10):
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
st.merge(method=1, fill_value='interpolate')
ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
ppsd.add(st)

target_f = 1.0
psd, times = extract_psd_time_series(ppsd, target_f, True, smooth=False)
psd = pd.DataFrame(psd, index=times, columns=["psd_db"]).sort_index()

uv = pd.read_csv("./checkpoints/tidal_currents.csv")

n = min(len(psd), len(uv))
psd = psd.iloc[:n]
uv  = uv.iloc[:n]

uv.index = psd.index

U_speed = np.sqrt(uv["u"].values**2 + uv["v"].values**2)/100

fs = np.array([1/14, 0.1, 1.0, 1.5, 5.0, 10.0])
df = pd.DataFrame(columns=fs)

for freq in fs:
    psd, times = extract_psd_time_series(ppsd, freq, True, smooth=False)
    psd = pd.DataFrame(psd, index=times, columns=["psd_db"]).sort_index()
    PSD = psd["psd_db"].values

    fU, Cxy_U = coherence(PSD, U_speed, fs=FS_CPD, nperseg=NPERSEG)
    
    df[freq] = Cxy_U
df.index = fU

df.to_csv('./checkpoints/node_coherence.csv', sep=',')
