# fetch_and_compare_tide_currents.py
import numpy as np
import pandas as pd
from scipy.signal import detrend
import matplotlib.pyplot as plt
from obspy import Stream, read, read_inventory
from obspy.signal import PPSD

from helper_funcs import *


t0 = '2025-04-01'
t1 = '2025-04-10'
dt_minutes = 30
psd_cadence = f'{dt_minutes}min'

station = '03479'
comp = 'Z'
st = Stream()
for d in range(1,10):
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
st.merge(method=1, fill_value='interpolate', interpolation_samples=10)
# st.detrend('linear')
# st.taper(max_percentage=0.05, type='cosine')
# st.filter('bandpass', freqmin=0.2, freqmax=5.0, corners=4, zerophase=True)
ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
ppsd.add(st)
psd_series, times = extract_psd_time_series(ppsd, 1, True, smooth=False)

tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
dt = tidal_df.index.to_series().diff().dt.total_seconds()
times = pd.to_datetime(times)
match_idx = tidal_df.index.isin(times)
tidal_df = tidal_df.loc[match_idx, :]
tidal_series = tidal_df['ASLVBG02'].reindex(psd_series.index)

tidal_h = pd.read_csv('./checkpoints/tidal_heights.csv', index_col=False, header=0, delimiter=',')['0'].to_numpy()
tidal_uv = pd.read_csv('./checkpoints/tidal_currents.csv', index_col=False, header=0, delimiter=',')
curr_speed = np.sqrt(tidal_uv['u']**2 + tidal_uv['v']**2)/100
if type(curr_speed) == pd.Series:   curr = curr_speed; curr.index = times
else:                               curr = pd.Series(curr_speed, index=times)

tide = tidal_series.reindex(times).interpolate()   # align and fill

# detrend and z-score
tide_z = zscore(detrend(tidal_h))
curr_z = zscore(detrend(curr.values))

tidal_h = pd.Series(tidal_h, index=times)
tide_z = pd.Series(tide_z, index=times)
curr_z = pd.Series(curr_z, index=times)
fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tidal_h, 
                                                        24, overlap_frac=0.75, show=False, var_label='Tidal height (m)', normalise_window=False)
fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tide_z, 
                                                        24, overlap_frac=0.75, show=False, var_label='Tidal height', normalise_window=False)

plt.show()
