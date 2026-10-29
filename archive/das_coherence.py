import numpy as np
import pandas as pd
from scipy.signal import coherence
from datetime import datetime, timedelta

from helper_funcs import *
from weather import *


FS_CPD = 48.0          # samples per day (30-min cadence -> 48 samples/day)
NPERSEG = 1008         # Welch segment length = 21 days

window_seconds = 30 * 60
half_window = window_seconds / 2.0


fs = np.array([1/50, 1/14, 0.1, 1.0, 1.5, 5, 10])
for cha in [750, 1475]:
    df = pd.DataFrame(columns=fs)
    t0 = datetime(year=2024, month=4, day=1); t1 = datetime(year=2025, month=7, day=31)

    lons = [1.402138]; lats = [52.94651]
    times = pd.date_range(f'{t0.date()} 00:00:00', f'{t1.date()} 23:30:00', freq=timedelta(minutes=30))
    if os.path.isfile(f'./checkpoints/tidal_heights_{t0}_{t1}.csv'):
        h = pd.read_csv(f'./checkpoints/tidal_heights_{t0}_{t1}.csv', index_col=False, header=0, delimiter=',', names=["H"])
        uv = pd.read_csv(f'./checkpoints/tidal_currents_{t0}_{t1}.csv', index_col=False, header=0, delimiter=',')
    else:
        h, uv = get_tidal_data(lons=lons, lats=lats, times=times)
        h = pd.DataFrame(h, columns=['H'])
        uv = pd.DataFrame(uv, columns=['u','v'])
        h.to_csv(f'./checkpoints/tidal_heights_{t0}_{t1}.csv', index=False)
        uv.to_csv(f'./checkpoints/tidal_currents_{t0}_{t1}.csv', index=False)

    files = get_spec_files(f'/data/localraid/saved_specs/600s_window/', t0, t1+timedelta(days=1), cha)
    for i, freq in enumerate(fs):
        cosine_30min = []
        for file_path, t_start in files: 
            spec = np.loadtxt(file_path, delimiter=',')
            f_size, t_size = spec.shape
            f_bins = np.linspace(0.0, 50.016666666666666, f_size)
            f_idx = (np.abs(f_bins - freq)).argmin()
            
            t_start = datetime.strptime(file_path.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S')
            t_end   = datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S')
            t_bin_length = (t_end - t_start).total_seconds() / t_size
            time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]
            
            psd_series = pd.Series(spec[f_idx,:], index=pd.to_datetime(time_axis))
            psd_series = psd_series[f'{t_start.date()} 00:00:00':f'{t_end.date()} 23:59:59']
            vals = psd_series.values
            idx = psd_series.index
            
            targets = pd.DatetimeIndex(pd.date_range(f'{t_start.date()} 00:00:00', f'{(t_end-timedelta(days=1)).date()} 23:30:00', freq=timedelta(minutes=30)))
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
        
        targets = pd.DatetimeIndex(pd.date_range(f'{t0.date()} 00:00:00', f'{(t_end-timedelta(days=1)).date()} 23:30:00', freq=timedelta(minutes=30)))
        psd_30min_cosine = pd.Series(cosine_30min, index=targets)
        
        U_speed = np.sqrt(uv["u"].values**2 + uv["v"].values**2)/100
        mask = ~np.isnan(psd_30min_cosine)
        psd_30min_cosine = psd_30min_cosine[mask]
        U_speed = U_speed[mask]
        
        PSD = psd_30min_cosine.values

        fU, Cxy_U = coherence(PSD, U_speed, fs=FS_CPD, nperseg=NPERSEG, noverlap=int(NPERSEG/2))
        df[freq] = Cxy_U
    df.index = fU

    df.to_csv(f'./checkpoints/das_coherence_{cha}.csv', sep=',')
