import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from helper_funcs import *
from weather import *


file_path = '/data/localraid/Fibre Data - HW/saved_specs/600s_window/2024-12-01 00:00:06__2025-01-01 00:00:06_f0.01:49.9_1475_spec.txt'
# file_path = '/data/localraid/saved_specs/non_months/600s_window/2025-04-01 00:00:25__2025-04-10 00:00:25_f0.01:49.9_1475_spec.txt'
spec = np.loadtxt(file_path, delimiter=',')
f_size, t_size = spec.shape

t_start = datetime.strptime(file_path.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S')
t_end   = datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S')
t_bin_length = (t_end - t_start).total_seconds() / t_size
time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]

target_f = 1.6
f_bins = np.linspace(0.0, 50.016666666666666, f_size)
idx = (np.abs(f_bins - target_f)).argmin()
psd_series = pd.Series(spec[idx,:], index=pd.to_datetime(time_axis))
psd_series = psd_series[f'{t_start.date()} 00:00:00':f'{t_end.date()} 23:59:59']

lons = [1.402138]; lats = [52.94651]
times = pd.date_range(f'{t_start.date()} 00:00:00', f'{(t_end-timedelta(days=1)).date()} 23:00:00', freq=timedelta(minutes=30))

df_wave = plot_waverider_csv('../NoisePy4DAS/results/checkpoints/hpg_wave.csv', False, get_df=True)
df_met = plot_met_csv('../NoisePy4DAS/results/checkpoints/hpg_met.csv', False, get_df=True)
df_era5 = plot_era5_csv('../NoisePy4DAS/results/checkpoints/combined_weather.csv', False, get_df=True)
weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[t_start.date():t_end.date()]
print(weather_df)
weather_df = weather_df.reindex(times)

tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
match_idx = tidal_df.index.isin(times)
tidal_df = tidal_df.loc[match_idx, :]
tidal_series = tidal_df['ASLVBG02'].reindex(times)

window_seconds = 30 * 60
half_window = window_seconds / 2.0
targets = pd.DatetimeIndex(times)
vals = psd_series.values
idx = psd_series.index

cosine_30min = []
for t in targets:
    offs = (idx - t).total_seconds()
    sel = np.abs(offs) <= half_window
    if not np.any(sel):
        cosine_30min.append(np.nan)
        continue
    offs_sel = offs[sel]
    samples = vals[sel].astype(float)
    w = 0.5 * (1.0 + np.cos(2.0 * np.pi * offs_sel / window_seconds))
    good = np.isfinite(samples) & (w > 0)
    if not np.any(good):
        cosine_30min.append(np.nan)
        continue
    w = w[good]
    s = samples[good]
    cosine_30min.append(np.sum(w * s) / np.sum(w))

psd_series = psd_series.interpolate()
psd_30min_cosine = pd.Series(cosine_30min, index=targets)
weather_df['Wind(m/s)'] = weather_df['Wind(m/s)'].interpolate()
weather_df['Hs(Hm0)(m)'] = weather_df['Hs(Hm0)(m)'].interpolate()
tide = tidal_series.reindex(times).interpolate()

fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, weather_df['Wind(m/s)'], window_hours=24, overlap_frac=0.75, show=True, var_label='Wind speed (m/s)', normalise_window=False)

plt.show()
