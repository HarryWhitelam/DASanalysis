import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from pyTMD.compute import tide_elevations, tide_currents

from weather import *


def zscore(x): return (x - np.nanmean(x)) / np.nanstd(x)


def extract_psd_time_series(ppsd, frequency_hz, return_times=False, smooth=False, mad_thresh=6.0, window_len=5):
    """
    Extract time series of PSD (dB, as stored in PPSD) for a single frequency (Hz).
    Returns a pandas.Series indexed by datetime.
    """

    target_period = 1.0 / float(frequency_hz)
    periods = np.asarray(ppsd.period_bin_centers)
    idx = int(np.argmin(np.abs(periods - target_period)))

    times_out = []
    vals_out = []

    for times_segment, psds_segment in ppsd._get_gapless_psd():
        if len(times_segment) == 0:
            continue
        arr = np.asarray(psds_segment)
        if arr.ndim != 2:
            arr = arr.reshape((len(times_segment), -1)).T if arr.size == len(periods) * len(times_segment) else arr
        if arr.shape[0] == len(times_segment) and arr.shape[1] == len(periods):
            vals = arr[:, idx]
        elif arr.shape[1] == len(times_segment) and arr.shape[0] == len(periods):
            vals = arr[idx, :]
        else:
            arrT = arr.T
            if arrT.shape[0] == len(times_segment) and arrT.shape[1] == len(periods):
                vals = arrT[:, idx]
            elif arrT.shape[1] == len(times_segment) and arrT.shape[0] == len(periods):
                vals = arrT[idx, :]
            else:
                raise RuntimeError("Unexpected psds_segment shape: %s" % (arr.shape,))

        times_out.extend([pd.Timestamp(t.datetime) for t in times_segment])
        vals_out.extend(list(vals))

    if len(times_out) == 0:
        if return_times:
            return pd.Series(dtype=float), np.array([], dtype='datetime64[ns]')
        return pd.Series(dtype=float)

    raw_index = pd.DatetimeIndex(times_out)
    raw_series = pd.Series(data=np.array(vals_out, dtype=float), index=raw_index).sort_index()

    if smooth:
        med = raw_series.rolling(window=window_len, center=True, min_periods=1).median()
        dev = (raw_series - med).abs()
        mad = dev.median()
        if mad == 0 or not np.isfinite(mad):
            mad = dev.std(ddof=0)
        if mad == 0 or not np.isfinite(mad):
            outliers = np.zeros_like(dev, dtype=bool)
        else:
            outliers = dev > (mad_thresh * mad)
        raw_series = raw_series.where(~outliers, med)
        raw_series = raw_series.interpolate(method='time', limit=3)
    series = raw_series.groupby(raw_series.index).mean()

    if return_times:
        raw_times = np.array(pd.DatetimeIndex(times_out).astype('datetime64[ns]'))
        return series, raw_times
    return series


def rolling_xcorr_plot(psd_series, tidal_series,
                       window_hours=24, overlap_frac=0.75,
                       normalise_window=True, cmap='seismic',
                       vmax=1.0, vmin=-1.0, show=True, var_label=None):
    """
    Compute rolling-window cross-correlation between two pandas.Series (same index).
    Returns (fig, ax, corr_matrix, lags, starts).

    - psd_series, tidal_series: pandas.Series aligned on the same DatetimeIndex.
    - window_hours: window length in hours (default 24).
    - overlap_frac: fraction overlap between windows (default 0.75).
    - normalise_window: normalise each window's correlation by its max abs value.
    - cmap, vmin, vmax: imshow colormap and limits.
    """

    common_idx = psd_series.index.intersection(tidal_series.index)
    ps = psd_series.reindex(common_idx); td = tidal_series.reindex(common_idx)
    mask = ps.notna() & ps.notna()
    ps = ps[mask]; td = td[mask]

    dt_sec = pd.Series(ps.index).diff().median().total_seconds()
    window_seconds = window_hours * 3600
    window_samps = max(2, int(round(window_seconds / dt_sec)))
    step = max(1, int(round(window_samps * (1.0 - overlap_frac))))
    if step == 0:
        step = 1

    N = len(ps)
    starts = np.arange(0, N - window_samps + 1, step)
    if starts.size == 0:
        raise RuntimeError("Window length larger than data span")

    n_windows = len(starts)
    corr_len = 2 * window_samps - 1
    corr_matrix = np.full((n_windows, corr_len), np.nan, dtype=float)
    lags = np.arange(-window_samps + 1, window_samps) * dt_sec

    psd_vals = ps.to_numpy(dtype=float)
    tidal_vals = td.to_numpy(dtype=float)
    var_window_mean = np.full(n_windows, np.nan, dtype=float)

    for i, s in enumerate(starts):
        x = psd_vals[s: s + window_samps]
        y = tidal_vals[s: s + window_samps]
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            continue
        if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
            continue
        xn = (x - x.mean()) / x.std(ddof=0)
        yn = (y - y.mean()) / y.std(ddof=0)
        corr = np.correlate(xn, yn, mode='full') / window_samps
        if normalise_window:
            m = np.max(np.abs(corr))
            norm_corr = corr if m == 0 else corr / m
        else:
            norm_corr = corr
        corr_matrix[i, :] = norm_corr
        var_window_mean[i] = np.nanmean(y)
        
    peak_lags = np.full(n_windows, np.nan, dtype=float)
    peak_vals = np.full(n_windows, np.nan, dtype=float)
    for i in range(n_windows):
        row = corr_matrix[i, :]
        if np.all(np.isnan(row)):
            continue
        peak_idx = int(np.nanargmax(np.abs(row)))
        peak_lags[i] = lags[peak_idx]
        peak_vals[i] = row[peak_idx]

    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(1, 2, width_ratios=[0.82, 0.18], wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_tidal = fig.add_subplot(gs[0, 1])

    extent = [lags[0], lags[-1], 0, n_windows]
    im = ax.imshow(corr_matrix, aspect='auto', extent=extent, origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Lag (hours)')
    lag_hours = lags / 3600.0
    max_abs_h = int(np.ceil(max(abs(lag_hours.min()), abs(lag_hours.max()))))
    tick_step = max(1, window_hours // 4)  # hours between ticks
    if max_abs_h >= tick_step:
        max_block = int(np.ceil(max_abs_h / tick_step)) * tick_step
        hours = np.arange(-max_block, max_block + 1, tick_step, dtype=int)
        hours = hours[(hours >= -max_abs_h) & (hours <= max_abs_h)]
    else:
        hours = np.arange(-max_abs_h, max_abs_h + 1, dtype=int)
    if 0 not in hours and (lag_hours.min() <= 0.0 <= lag_hours.max()):
        hours = np.sort(np.concatenate((hours, np.array([0], dtype=int))))
    xt_pos = hours * 3600.0
    xt_labels = [f"{h}h" for h in hours]
    ax.set_xticks(xt_pos)
    ax.set_xticklabels(xt_labels)
    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Window index')

    y_centres = np.arange(n_windows) + 0.5
    valid = ~np.isnan(peak_lags)
    if np.any(valid):
        ax.scatter(peak_lags[valid], y_centres[valid], c='k', s=8, zorder=5)
    
    mid_indices = (starts + window_samps // 2).astype(int)
    centre_times = pd.DatetimeIndex(ps.index[mid_indices])
    # preferred_hours = [0, 12]
    preferred_hours = [0]
    tick_idxs = np.where(np.isin(centre_times.hour, preferred_hours))[0]

    if tick_idxs.size == 0:
        nlabels = min(8, n_windows)
        tick_idxs = np.linspace(0, n_windows - 1, nlabels).astype(int)

    max_ticks = 12
    if tick_idxs.size > max_ticks:
        step = int(np.ceil(tick_idxs.size / max_ticks))
        tick_idxs = tick_idxs[::step]

    yt_pos = tick_idxs + 0.5
    yt_labels = [centre_times[i].strftime('%Y-%m-%d\n%H:%M') for i in tick_idxs]
    ax.set_yticks(yt_pos)
    ax.set_yticklabels(yt_labels)

    # plot the full tidal series vertically aligned with the main xcorr y-axis
    # centre_times / y_centres define the mapping from time -> window index (y)
    y_centres = np.arange(n_windows) + 0.5
    centre_nums = centre_times.astype('int64')        # ns since epoch
    # use the reindexed tidal series (td) so it matches ps index used for windows
    tidal_idx_nums = td.index.astype('int64')
    tidal_vals_aligned = td.to_numpy(dtype=float)

    # interpolate each tidal timestamp onto the window-index y axis
    y_map = np.full_like(tidal_idx_nums, np.nan, dtype=float)
    in_range = (tidal_idx_nums >= centre_nums[0]) & (tidal_idx_nums <= centre_nums[-1])
    if np.any(in_range):
        y_map[in_range] = np.interp(tidal_idx_nums[in_range], centre_nums, y_centres)

    # plot only finite points
    good = np.isfinite(y_map) & np.isfinite(tidal_vals_aligned)
    if np.any(good):
        ax_tidal.plot(tidal_vals_aligned[good], y_map[good], color='C0', lw=0.7)

    # overlay per-window mean (for reference)
    ax_tidal.plot(var_window_mean, y_centres, 'ko', markersize=3, label='window mean')
    ax_tidal.set_xlabel(f'{var_label}')
    ax_tidal.set_xlim(np.nanmin(tidal_vals_aligned[good]) if np.any(good) else 0,
                      np.nanmax(tidal_vals_aligned[good]) if np.any(good) else 1)
    ax_tidal.set_ylim(0.0, n_windows)
    # show same y ticks/labels as main plot so times line up visually
    ax_tidal.set_yticks(yt_pos)
    ax_tidal.set_yticklabels([])
    ax_tidal.grid()

    cbar = fig.colorbar(im, ax=ax_tidal, label='Normalised correlation coefficient' if normalise_window else 'Correlation coefficient', fraction=0.046, pad=0.01)
    ax.grid()
    # fig.suptitle(f'Rolling {window_hours}h cross-correlation ({n_windows} windows, dt={dt_sec:.1f}s)')
    # plt.tight_layout()
    if show:
        plt.show()

    return fig, ax, corr_matrix, lags, starts           


def get_tidal_data(lats, lons, times):
    tide_h = tide_elevations(
        x=lons,
        y=lats,
        delta_time=times,
        DIRECTORY="/data/localraid/tide_models/",
        MODEL="FES2022",
        EPSG=4326,
        TIME="datetime",
    )
    tide_uv = tide_currents(
        x=lons,
        y=lats,
        delta_time=times,
        DIRECTORY="/data/localraid/tide_models/",
        MODEL="FES2014",
        EPSG=4326,
        TIME="datetime",
    )
    return tide_h, tide_uv


def get_spec_files(dir:str, t0:datetime, t1:datetime, target_cha:int):
    files = []
    for dir_path, dir_names, file_names in os.walk(dir):
        for file in file_names:
            try:
                f_split = file.split('_')
                t_start = datetime.strptime(f_split[0], '%Y-%m-%d %H:%M:%S')
                t_end = datetime.strptime(f_split[2], '%Y-%m-%d %H:%M:%S')
                if target_cha and int(f_split[-2]) != target_cha: continue
                if t0 and t_start.date() < t0.date(): continue
                if t1 and t_end.date() > t1.date(): continue
            except Exception: 
                continue        # basically there was no datetime within the file
            files.append((os.path.join(dir_path, file), t_start))
    files.sort(key=lambda x: x[1])
    return files


def get_weather_data(t_start, t_end, times, cols=None):
    df_wave = plot_waverider_csv('../NoisePy4DAS/results/checkpoints/hpg_wave.csv', False, get_df=True)
    df_met = plot_met_csv('../NoisePy4DAS/results/checkpoints/hpg_met.csv', False, get_df=True)
    df_era5 = plot_era5_csv('../NoisePy4DAS/results/checkpoints/combined_weather.csv', False, get_df=True)
    weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[t_start.date():t_end.date()]
    weather_df = weather_df.reindex(times)
    weather_df['V(m/s)'] = weather_df['V(m/s)'].interpolate()
    weather_df['Wind(m/s)'] = weather_df['Wind(m/s)'].fillna(weather_df['V(m/s)'])

    tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
    match_idx = tidal_df.index.isin(times)
    tidal_df = tidal_df.loc[match_idx, :]
    weather_df['ASLVBG02'] = tidal_df['ASLVBG02'].reindex(times)
    
    if 'curr' in cols:
        lons = [1.402138]; lats = [52.94651]
        tidal_h, tidal_uv = get_tidal_data(lons=lons, lats=lats, times=times)
        curr = np.sqrt(tidal_uv['u']**2 + tidal_uv['v']**2)
        curr = pd.Series(np.asarray(curr), index=times).interpolate()
        weather_df['curr'] = curr
    
    if cols:
        weather_df = weather_df[cols]
    return weather_df
