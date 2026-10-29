import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed

from helper_funcs import get_spec_files, get_weather_data

corr_type = 'spearman'
# fs = [0.1, 0.3, 0.5, 1.0, 1.6, 2.0, 5.0, 10.0]
fs = [0.1, 0.3, 0.5, 1.0, 1.5, 1.6, 2.0, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]
chas = [750, 788, 875, 1475]
t0 = datetime(year=2024, month=4, day=1)
t1 = datetime(year=2025, month=8, day=1)

files = get_spec_files('/data/localraid/saved_specs/600s_window/', t0, t1, None)
months = sorted(set(file[1] for file in files))
months = [month for month in months if month.day == 1]
month_idxs = [month.date() for month in months]
cols = ['corr_full', 'block_mean', 'block_std', 'boot_lo', 'boot_hi', 'n_blocks', 'n_samples']

wind_df = pd.DataFrame(index=pd.MultiIndex.from_product([chas, fs, month_idxs], names=('channels', 'frequencies', 'months')), columns=cols)
wh_df = pd.DataFrame(index=pd.MultiIndex.from_product([chas, fs, month_idxs], names=('channels', 'frequencies', 'months')), columns=cols)
curr_df = pd.DataFrame(index=pd.MultiIndex.from_product([chas, fs, month_idxs], names=('channels', 'frequencies', 'months')), columns=cols)
tidalh_df = pd.DataFrame(index=pd.MultiIndex.from_product([chas, fs, month_idxs], names=('channels', 'frequencies', 'months')), columns=cols)
dfs = [wind_df, wh_df, tidalh_df, curr_df]

block_bootstrap_method = 'resample_blocks'  # or 'moving_blocks'
SAMPLES_PER_DAY = 48
boot_B = 1000

target_cols = ['Wind(m/s)','Hs(Hm0)(m)','ASLVBG02','curr']
weather_t_start = months[0]
weather_t_end = months[-1] + relativedelta(months=1)
weather_times = pd.date_range(f'{weather_t_start.date()} 00:00:00', f'{weather_t_end.date()} 00:00:00', freq=timedelta(minutes=30))
full_weather_df = get_weather_data(weather_t_start, weather_t_end, weather_times, target_cols)


def process_month(t_start, cha, seed=0):
    """Process one month (t_start) and channel (cha). Return list of (key, stats_dict)"""
    rng_local = np.random.default_rng(seed)
    out = []
    t_end = t_start + relativedelta(months=1)
    times = pd.date_range(f'{t_start.date()} 00:00:00', f'{t_end.date()} 00:00:00', freq=timedelta(minutes=30))
    weather_df = full_weather_df[t_start.date():t_end.date()]

    # load spec file
    file_path = f'/data/localraid/saved_specs/600s_window/{t_start}__{t_end}_f0.01:49.9_{cha}_spec.txt'
    try:
        spec = np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        # return empty if file missing
        return out
    f_size, t_size = spec.shape
    t_bin_length = (t_end - t_start).total_seconds() / t_size
    time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]
    f_bins = np.linspace(0.0, 50.016666666666666, f_size)

    # compute psd_30min_cosine for each target frequency and then correlations per target_col
    for target_f in fs:
        idx_f = (np.abs(f_bins - target_f)).argmin()
        psd_series = pd.Series(spec[idx_f, :], index=pd.to_datetime(time_axis))
        psd_series = psd_series[f'{t_start.date()} 00:00:00':f'{t_end.date()} 23:59:59']

        # resample/aggregate to the 30-min targets using the cosine window (same logic as before)
        window_seconds = 30 * 60
        half_window = window_seconds / 2.0
        targets = pd.DatetimeIndex(times)
        vals = psd_series.values
        idx_ts = psd_series.index
        cosine_30min = []
        for t in targets:
            offs = (idx_ts - t).total_seconds()
            sel = np.abs(offs) <= half_window
            if not np.any(sel):
                cosine_30min.append(np.nan); continue
            offs_sel = offs[sel]
            samples = vals[sel].astype(float)
            w = 0.5 * (1.0 + np.cos(2.0 * np.pi * offs_sel / window_seconds))
            good = np.isfinite(samples) & (w > 0)
            if not np.any(good):
                cosine_30min.append(np.nan); continue
            w = w[good]; s = samples[good]
            cosine_30min.append(np.sum(w * s) / np.sum(w))
        psd_30min_cosine = pd.Series(cosine_30min, index=targets)

        # sliding blocks
        window_days = 7; step_days = 1
        last_start = t_end - pd.Timedelta(days=window_days)
        if last_start < t_start:
            window_starts = pd.DatetimeIndex([t_start])
        else:
            window_starts = pd.date_range(start=t_start, end=last_start, freq=f'{step_days}D')

        # compute stats for every target weather column
        for col, df_name in zip(target_cols, ['wind','wh','tidalh','curr']):
            # initialize stats container
            stats_out = {'corr_full': np.nan, 'block_mean': np.nan, 'block_std': np.nan,
                         'boot_lo': np.nan, 'boot_hi': np.nan, 'n_blocks': 0, 'n_samples': 0}

            # full-month correlation (align automatically via pandas)
            pair_full = pd.concat([psd_30min_cosine, weather_df[col]], axis=1).dropna()
            if pair_full.shape[0] >= 4:
                xi_full = pair_full.iloc[:,0].to_numpy(dtype=float)
                yi_full = pair_full.iloc[:,1].to_numpy(dtype=float)
                if corr_type == 'pearson':
                    stats_out['corr_full'] = np.corrcoef(xi_full, yi_full)[0,1]
                else:
                    stats_out['corr_full'] = stats.spearmanr(xi_full, yi_full).correlation
            # sliding block correlations
            block_rs = []
            block_ns = []
            block_pairs = []
            for ws in window_starts:
                we = ws + pd.Timedelta(days=window_days) - pd.Timedelta(seconds=1)
                pair = pd.concat([psd_30min_cosine.loc[ws:we], weather_df[col].loc[ws:we]], axis=1).dropna()
                n = pair.shape[0]
                block_ns.append(n)
                if n < 4:
                    block_rs.append(np.nan); block_pairs.append(None); continue
                xi = pair.iloc[:,0].to_numpy(dtype=float); yi = pair.iloc[:,1].to_numpy(dtype=float)
                if corr_type == 'pearson':
                    r = np.corrcoef(xi, yi)[0,1]
                else:
                    r = stats.spearmanr(xi, yi).correlation
                block_rs.append(r); block_pairs.append((xi, yi))

            block_rs = np.asarray(block_rs, dtype=float)
            stats_out['block_mean'] = np.nanmean(block_rs) if np.any(np.isfinite(block_rs)) else np.nan
            stats_out['block_std'] = np.nanstd(block_rs, ddof=1) if np.sum(np.isfinite(block_rs)) > 1 else np.nan
            stats_out['n_blocks'] = int(np.sum(np.isfinite(block_rs)))
            stats_out['n_samples'] = int(np.nansum(block_ns))

            # block-bootstrap (resample raw block pairs) — faster cheap mode: resample block correlations
            valid_pairs = [p for p in block_pairs if p is not None]
            n_blocks = len(valid_pairs)
            if n_blocks >= 2:
                if block_bootstrap_method == 'resample_blocks':
                    boot_means = np.empty(boot_B, dtype=float)
                    for b in range(boot_B):
                        idxs = rng_local.integers(0, n_blocks, n_blocks)
                        rvals = []
                        for ii in idxs:
                            xa, ya = valid_pairs[ii]
                            if corr_type == 'pearson':
                                rv = np.corrcoef(xa, ya)[0,1]
                            else:
                                rv = stats.spearmanr(xa, ya).correlation
                            rvals.append(rv)
                        boot_means[b] = np.nanmean(rvals)
                    stats_out['boot_lo'], stats_out['boot_hi'] = np.nanpercentile(boot_means, [2.5, 97.5])
                elif block_bootstrap_method == 'moving_blocks':
                    # assemble full-length bootstrap via sampled contiguous blocks (in samples)
                    N = len(psd_30min_cosine)
                    block_size = max(1, int(window_days * SAMPLES_PER_DAY))
                    if N >= 2 and block_size >= 1:
                        boot_corrs = np.empty(boot_B, dtype=float)
                        psd_arr = psd_30min_cosine.to_numpy(dtype=float)
                        weather_arr = weather_df[col].to_numpy(dtype=float)
                        max_start = max(0, N - block_size)
                        n_blocks_needed = int(np.ceil(N / block_size))
                        for b in range(boot_B):
                            xb_parts = []; yb_parts = []
                            for k in range(n_blocks_needed):
                                s = rng_local.integers(0, max_start + 1)
                                xb_parts.append(psd_arr[s:s+block_size]); yb_parts.append(weather_arr[s:s+block_size])
                            xb_full = np.concatenate(xb_parts)[:N]; yb_full = np.concatenate(yb_parts)[:N]
                            mask = np.isfinite(xb_full) & np.isfinite(yb_full)
                            if mask.sum() < 4:
                                boot_corrs[b] = np.nan
                            else:
                                if corr_type == 'pearson':
                                    boot_corrs[b] = np.corrcoef(xb_full[mask], yb_full[mask])[0,1]
                                else:
                                    boot_corrs[b] = stats.spearmanr(xb_full[mask], yb_full[mask]).correlation
                        stats_out['boot_lo'], stats_out['boot_hi'] = np.nanpercentile(boot_corrs, [2.5, 97.5])
            # append result for this column
            out.append(((cha, target_f, t_start.date()), col, stats_out))
    return out


# --- run in parallel and collect results ---
tasks = [(m, c) for m in months for c in chas]
results = Parallel(n_jobs=-1, verbose=5)(delayed(process_month)(m, c, seed=42 + i) for i, (m, c) in enumerate(tasks))

# flatten and write into DataFrames
for sublist in results:
    for key, col, stats_out in sublist:
        # map col -> df
        df = wind_df if col == 'Wind(m/s)' else (wh_df if col == 'Hs(Hm0)(m)' else (tidalh_df if col == 'ASLVBG02' else curr_df))
        df.loc[key, 'corr_full'] = stats_out['corr_full']
        df.loc[key, 'block_mean'] = stats_out['block_mean']
        df.loc[key, 'block_std'] = stats_out['block_std']
        df.loc[key, 'boot_lo'] = stats_out['boot_lo']
        df.loc[key, 'boot_hi'] = stats_out['boot_hi']
        df.loc[key, 'n_blocks'] = stats_out['n_blocks']
        df.loc[key, 'n_samples'] = stats_out['n_samples']

wind_df.to_csv(f'/data/localraid/weather_corrs/full_wind_corr_{corr_type}.csv')
wh_df.to_csv(f'/data/localraid/weather_corrs/full_wh_corr_{corr_type}.csv')
curr_df.to_csv(f'/data/localraid/weather_corrs/full_curr_corr_{corr_type}.csv')
tidalh_df.to_csv(f'/data/localraid/weather_corrs/full_tidalh_corr_{corr_type}.csv')


cha = 750; freq = 0.5
sub = wind_df.loc[(cha, freq)]
x = pd.to_datetime(sub.index)
y = sub['block_mean'].astype(float).values
yerr = sub['block_std'].astype(float).values
# SE from bootstrap interval if available:
lo = sub['boot_lo'].astype(float).values
hi = sub['boot_hi'].astype(float).values
y_se = np.where(np.isfinite(lo) & np.isfinite(hi), (hi - lo) / (2*1.96), np.nan)

fig, ax = plt.subplots(figsize=(8,3.5))
ax.errorbar(x, y, yerr=yerr, fmt='o-', label='block mean ± std')
ax.fill_between(x, y - 1.96*y_se, y + 1.96*y_se, alpha=0.15, label='bootstrap 95% CI (approx)')
ax.set_title(f'Cha {cha} @ {freq} Hz')
ax.legend(); plt.show()
