from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.signal import PPSD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence
from datetime import timedelta
from helper_funcs import *
from weather import *


### xcorr of tidal data and spectra
station = '04915'   # 03479
comp = 'E'

st = Stream()
for d in range(1,10):
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
st.merge(method=1, fill_value='interpolate')
# st.detrend('linear')
# st.taper(max_percentage=0.05, type='cosine')
# st.filter('bandpass', freqmin=0.2, freqmax=5.0, corners=4, zerophase=True)
ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
ppsd.add(st)
psd_series, times = extract_psd_time_series(ppsd, 1.0, True, smooth=False)
# psd_series.to_csv('/checkpoints/psd_1hz.csv')

tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
dt = tidal_df.index.to_series().diff().dt.total_seconds()
tidal_df['derivative'] = tidal_df['ASLVBG02'].diff() / dt
tidal_df['derivative_2'] = tidal_df['derivative'].diff() / dt
times = pd.to_datetime(times)
match_idx = tidal_df.index.isin(times)
tidal_df = tidal_df.loc[match_idx, :]

df_wave = plot_waverider_csv('../NoisePy4DAS/results/checkpoints/hpg_wave.csv', False, get_df=True)
df_met = plot_met_csv('../NoisePy4DAS/results/checkpoints/hpg_met.csv', False, get_df=True)
df_era5 = plot_era5_csv('../NoisePy4DAS/results/checkpoints/combined_weather.csv', False, get_df=True)
weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[times[0].date():times[-1].date()]
weather_df = weather_df.reindex(times)
weather_df['Hs(Hm0)(m)'] = weather_df['Hs(Hm0)(m)'].interpolate()

current_df = pd.read_csv('/local/gfs19eku/Documents/Nodes_Analysis/checkpoints/tidal_currents.csv')
curr_speed = np.sqrt(current_df['u'].to_numpy()**2 + current_df['v'].to_numpy()**2)
curr = pd.Series(curr_speed, index=times) / 100

delta = times[-1] - times[0]
ts_delta = delta.total_seconds() / (psd_series.shape[0]-1)
tidal_delta = delta.total_seconds() / (tidal_df.shape[0]-1)
if ts_delta != tidal_delta: print(f'deltas do not match! ts: {ts_delta} || tidal: {tidal_delta}')


tidal_series = zscore(tidal_df['ASLVBG02'].reindex(psd_series.index))
tidal_d_series = zscore(tidal_df['derivative'].reindex(psd_series.index))
tidal_series_norm = tidal_series.abs()
tidal_d_series_norm = tidal_d_series.abs()


fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, curr, 
                                                        24, overlap_frac=0.75, show=True, var_label='Tidal current (m/s)', normalise_window=False)
fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tidal_series_norm, 
                                                        24, overlap_frac=0.75, show=True, var_label='Tidal height (m)', normalise_window=False)
