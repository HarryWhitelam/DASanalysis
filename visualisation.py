import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import cupy as cp
from scipy.signal import welch, ShortTimeFFT, convolve2d, savgol_filter
from scipy.signal.windows import hamming
from obspy.imaging.util import ObsPyAutoDateFormatter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.rrule import MINUTELY, SECONDLY
import matplotlib
# matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cartopy.io.img_tiles as cimgt
from math import sin, cos, atan2, degrees, radians
import multiprocessing
from tqdm import tqdm

from TDMS_Read import TdmsReader
from tdms_io import get_filepath_array, get_data_from_array
from weather import plot_waverider_csv, plot_met_csv, plot_era5_csv


def octave_smooth(psd, freqs, width_octaves=0.5, axis=0, eps=1e-12):
    """
    ObsPy-like period smoothing: weighted average in linear power with weights
    proportional to period-bin widths (so smoothing conserves integrated power
    and avoids the large positive bias from naive arithmetic linear averaging).
    - psd: PSD in dB (shape with frequency on `axis`)
    - freqs: 1D frequency vector (Hz)
    - width_octaves: smoothing window width in octaves (default 0.5)
    - axis: frequency axis of psd
    - eps: small floor for numerical stability
    Returns: smoothed PSD in dB (same shape as input)
    """
    freqs = np.asarray(freqs, dtype=float)
    # guard DC / zero-frequency
    if freqs.size > 1 and freqs[0] == 0.0:
        freqs[0] = freqs[1] * 1e-6

    # periods and their log10 (period domain smoothing)
    periods = 1.0 / freqs
    logp = np.log10(periods)

    width_decades = width_octaves * np.log10(2.0)
    half_width = width_decades / 2.0

    # move frequency axis to axis 0
    psd_arr = np.asarray(psd)
    moved = np.moveaxis(psd_arr, axis, 0)
    n_freq = moved.shape[0]
    rest_shape = moved.shape[1:]
    n_cols = int(np.prod(rest_shape)) if rest_shape != () else 1
    flat = moved.reshape(n_freq, n_cols)  # shape (n_freq, n_cols)

    # linear power
    linear = 10.0 ** (flat / 10.0)

    # prepare sorted period axis for fast selection
    order = np.argsort(logp)
    logp_sorted = logp[order]
    periods_sorted = periods[order]

    # compute approximate period-bin widths (linear domain) for weighting
    # central difference for interior, forward/backward for ends
    widths = np.empty_like(periods_sorted)
    if periods_sorted.size > 1:
        widths[0] = periods_sorted[1] - periods_sorted[0]
        widths[-1] = periods_sorted[-1] - periods_sorted[-2]
        if periods_sorted.size > 2:
            widths[1:-1] = 0.5 * (periods_sorted[2:] - periods_sorted[:-2])
    else:
        widths[:] = 1.0

    # weighted average in linear domain using period widths as weights
    smoothed_linear = np.empty_like(linear)
    import bisect
    for i in range(n_freq):
        center = logp[i]
        left = center - half_width
        right = center + half_width
        i0 = bisect.bisect_left(logp_sorted, left)
        i1 = bisect.bisect_right(logp_sorted, right)
        sel = order[i0:i1]
        if sel.size == 0:
            smoothed_linear[i, :] = linear[i, :]
        else:
            sel_idx = slice(i0, i1)
            w = widths[sel_idx][:, None]             # shape (n_sel, 1)
            vals = linear[sel, :]                    # shape (n_sel, n_cols)
            weighted = (vals * w).sum(axis=0)
            wsum = w.sum()
            if wsum == 0:
                smoothed_linear[i, :] = linear[i, :]
            else:
                smoothed_linear[i, :] = weighted / wsum

    # convert back to dB and restore original shape/axis
    smoothed_db_flat = 10.0 * np.log10(smoothed_linear + eps)
    smoothed = smoothed_db_flat.reshape((n_freq,) + rest_shape)
    smoothed = np.moveaxis(smoothed, 0, axis)
    return smoothed


def octave_smooth_gpu(psd, freqs, width_octaves=0.5, axis=0, eps=1e-12, chunk_size_cols=512):
    """
    GPU-accelerated octave smoothing in column blocks to limit GPU memory.
    - psd: PSD in dB (numpy or cupy array)
    - freqs: 1D frequency vector (Hz)
    - chunk_size_cols: number of time/column slices to process per GPU transfer
    """
    # prepare frequency arrays on GPU
    freqs_cp = cp.asarray(freqs, dtype=cp.float64)
    if freqs_cp.size > 1 and freqs_cp[0] == 0.0:
        freqs_cp[0] = freqs_cp[1] * 1e-6
    periods = 1.0 / freqs_cp
    logp = cp.log10(periods)
    width_decades = width_octaves * cp.log10(2.0)
    half_width = width_decades / 2.0

    # prepare indexing/sorting arrays on GPU (small)
    order = cp.argsort(logp)
    logp_sorted = logp[order]
    periods_sorted = periods[order]

    # compute widths (period-bin widths) on GPU
    widths = cp.empty_like(periods_sorted)
    if periods_sorted.size > 1:
        widths[0] = periods_sorted[1] - periods_sorted[0]
        widths[-1] = periods_sorted[-1] - periods_sorted[-2]
        if periods_sorted.size > 2:
            widths[1:-1] = 0.5 * (periods_sorted[2:] - periods_sorted[:-2])
    else:
        widths[:] = 1.0
    # cumsum of widths (on GPU) - used later, length n_freq+1 with leading zero
    csum_w = cp.concatenate((cp.array([0.0], dtype=widths.dtype), cp.cumsum(widths)))

    # move frequency axis to axis 0 and compute shapes WITHOUT copying whole array to GPU
    is_cupy_input = isinstance(psd, cp.ndarray)
    if is_cupy_input:
        moved = cp.moveaxis(psd, axis, 0)
        n_freq = moved.shape[0]
        rest_shape = moved.shape[1:]
        n_cols = int(cp.prod(cp.asarray(rest_shape)).item()) if rest_shape != () else 1
    else:
        psd_arr = np.asarray(psd)
        moved = np.moveaxis(psd_arr, axis, 0)   # still CPU
        n_freq = moved.shape[0]
        rest_shape = moved.shape[1:]
        n_cols = int(np.prod(rest_shape)) if rest_shape != () else 1

    # prepare result flat array (host or device)
    result_flat = np.empty((n_freq, n_cols), dtype=np.float64)

    # prepare repeated index arrays on GPU (i0,i1) computed from logp_sorted
    lefts = logp - half_width
    rights = logp + half_width
    i0 = cp.searchsorted(logp_sorted, lefts, side='left')   # (n_freq,)
    i1 = cp.searchsorted(logp_sorted, rights, side='right')  # (n_freq,)

    # helper small GPU arrays that don't change per block
    order_cp = order
    widths_cp = widths
    csum_w_cp = csum_w
    # zeros row creation for cumsum per block done later (block dependent)

    # iterate column blocks
    for c0 in range(0, n_cols, chunk_size_cols):
        c1 = min(n_cols, c0 + chunk_size_cols)
        block_cols = c1 - c0

        # get block flat on GPU
        if is_cupy_input:
            # moved may be 1D (n_freq,) for single-column input; treat temporarily as (n_freq,1)
            if moved.ndim == 1:
                flat_block = moved[:, None][:, c0:c1]
            else:
                flat_block = moved[:, c0:c1]            # cupy slice, no extra copy if already on device
            flat_block_cp = flat_block.astype(cp.float64, copy=False)
        else:
            if moved.ndim == 1:
                flat_block = moved[:, None][:, c0:c1]
            else:
                flat_block = moved[:, c0:c1]            # numpy slice
            flat_block_cp = cp.asarray(flat_block, dtype=cp.float64)

        # linear power and sorted weighted cumsum (GPU)
        linear_block = 10.0 ** (flat_block_cp / 10.0)       # (n_freq, block_cols)
        linear_sorted = linear_block[order_cp, :]           # (n_freq, block_cols)
        w = widths_cp[:, None]                              # (n_freq, 1)
        weighted_vals = linear_sorted * w                   # (n_freq, block_cols)

        # prefix cumsum with zero-row (pad) - shapes (n_freq+1, block_cols)
        zero_row = cp.zeros((1, block_cols), dtype=weighted_vals.dtype)
        csum_vals = cp.concatenate((zero_row, cp.cumsum(weighted_vals, axis=0)), axis=0)

        # interval sums via indexing with i0,i1 (both length n_freq)
        # csum_vals has shape (n_freq+1, block_cols) and supports fancy indexing
        num = csum_vals[i1] - csum_vals[i0]   # (n_freq, block_cols)
        den = (csum_w_cp[i1] - csum_w_cp[i0])  # (n_freq,)

        # compute smoothed linear for this block
        smoothed_linear_block = cp.empty_like(flat_block_cp)
        mask_nonzero = den > 0.0
        if mask_nonzero.any():
            smoothed_linear_block[mask_nonzero, :] = num[mask_nonzero, :] / den[mask_nonzero, None]
        if (~mask_nonzero).any():
            smoothed_linear_block[~mask_nonzero, :] = linear_block[~mask_nonzero, :]

        # back to dB
        smoothed_db_block = 10.0 * cp.log10(smoothed_linear_block + eps)

        # write block to host
        result_flat[:, c0:c1] = cp.asnumpy(smoothed_db_block)

        # free block GPU memory promptly
        del flat_block_cp, linear_block, linear_sorted, weighted_vals, csum_vals, num, smoothed_linear_block, smoothed_db_block
        cp._default_memory_pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # reshape back to original shape and move axis back to original
    reshaped = result_flat.reshape((n_freq,) + rest_shape)
    smoothed = np.moveaxis(reshaped, 0, axis)
    del reshaped
    gc.collect()
    return smoothed


def parallel_spectral_analysis():
    dir_path = '/data/QNAP1_Data/Data/'
    
    for y, m, d, ds in [[2024, 8, 1, 1], [2025, 4, 1, 13560], [2024, 12, 1, None], [2024, 12, 8, 3], [2024, 12, 11, 3]]:
        t_start = datetime(year=y, month=m, day=d)
        if ds == None:
            t_end = t_start + relativedelta(months=1)
        elif ds == 13560:
            t_end = t_start + relativedelta(minutes=ds)
        else:
            t_end = t_start + relativedelta(days=ds)
        n_minutes = (t_end - t_start).total_seconds() / 60
        
        channels = [750, 788, 875, 1475]
        args_list = []
        prepro_para = {
            'samp_freq': 100,
            'target_spatial_res': 1,
            'n_minute': n_minutes,
            'freqmin': 0.01,
            'freqmax': 50.0,
        }
        
        filepath_array, timestamps = get_filepath_array(dir_path, t_start, t_end)
        t_start = timestamps[0].replace(microsecond=0)
        print(f'Data running from {t_start} to {timestamps[-1].replace(microsecond=0)}')
        data = get_data_from_array(filepath_array, prepro_para, t_start, duration=timedelta(minutes=n_minutes), channels=channels, filter=False)
    
        for i, channel in enumerate(channels):
            run_prepro_para = prepro_para.copy()
            run_prepro_para.update({'cha1':channel,
                                    'cha2':channel+1})
            channel_data = data[:,i]
            #                 dir_path, prepro_para, t_start, save_spec, data, window_length, plot_tides
            #                 dir_path, prepro_para, t_start, save_ppsd, data, window_length
            args_list.append((dir_path, run_prepro_para, t_start, False, channel_data))
    
        # p = multiprocessing.Pool(multiprocessing.cpu_count())
        # with tqdm(total=len(args_list), desc=f"{dir_path} spectrograms", position=0) as pbar:
        #     for _ in p.starmap(ts_spectrogram, args_list, chunksize=1):
        #         pbar.update(1)
        # p.close()
        
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        with tqdm(total=len(args_list), desc=f"{dir_path} PPSDs", position=0) as pbar:
            for _ in p.starmap(ppsd, args_list, chunksize=1):
                pbar.update(1)
        p.close()


def ts_spectrogram(dir_path:str, prepro_para:dict, t_start:datetime, save_spec=False, data=None, window_length=600, plot_tides=None):
    cha1, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('samp_freq'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    out_dir = f"./results/figures/PSD_Experiments/{window_length}s_window/"
    # out_dir = f"./results/figures/PSD_Experiments/smoothing_check/"
    
    if type(data)==type(None):
        filepath_array, timestamps = get_filepath_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(filepath_array, prepro_para, t_start, duration=timedelta(minutes=n_minute))[:, 0]
    
    N = data.shape[0]
    win = hamming(int(sps*window_length), sym=True)               # 28/10/2025 longer window, longer time averaging! Output name changed!
    # g_std = 12
    # gaussian_win = gaussian(sps*60, g_std, sym=True)
    stft = ShortTimeFFT(win, hop=int(sps*(window_length*0.5)), fs=sps, scale_to='psd')
    spec = stft.spectrogram(data)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = fig.add_subplot(111)
    ax.set_title(rf"DAS Channel {cha1}")
    spec = 10 * np.log10(spec + 1e-12)
    try:
        f_bins = np.linspace(freqmin, freqmax, stft.f_pts)
        spec = octave_smooth(spec, f_bins)
    except:
        print(f'octave smoothing failed!!!')
    
    # ext = stft.extent(N)
    ext = list(stft.extent(N))
    stft_duration = ext[1] - ext[0]
    ext[0] = mdates.date2num(t_start)
    ext[1] = mdates.date2num(t_start + timedelta(seconds=stft_duration))
    im1 = ax.imshow(spec, origin='lower', aspect='auto', 
                     extent=ext, cmap='jet', vmin=np.nanpercentile(spec,1), vmax=np.nanpercentile(spec,99))
    ax.set_yscale('log')
    plt.grid(which='both')
    
    if plot_tides:
        ax1 = ax.twinx()
        tidal_df = pd.read_csv('./results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
        tidal_df = tidal_df[t_start:t_start+timedelta(minutes=n_minute)]
        ax1.plot([(ts - t_start).total_seconds() for ts in tidal_df.index], tidal_df['ASLVBG02'])
        ax1.set_ylabel('Tidal Height (m)')
        plt.sca(ax=ax)
    
    if n_minute > 1440:
        ax.xaxis_date()
        locator = AutoDateLocator(minticks=3, maxticks=6)
        locator.intervald[MINUTELY] = [1, 2, 5, 10, 15, 30]
        locator.intervald[SECONDLY] = [1, 2, 5, 10, 15, 30]
        ax.xaxis.set_major_formatter(ObsPyAutoDateFormatter(locator))
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    else: 
        _ = plt.xticks(np.linspace(0, ext[1], 4), pd.date_range(t_start, t_start+timedelta(minutes=n_minute), periods=4), rotation=30)
        _ = plt.xticks(np.linspace(0, ext[1], 16), minor=True)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(freqmin, freqmax)
    
    fig.colorbar(im1, label=r'Amplitude [$n\epsilon^2/Hz$] [dB]')
    plt.subplots_adjust(top=0.922, bottom=0.15, left=0.131, right=0.953, hspace=0.2, wspace=0.2)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}_{freqmax}_{cha1}_spectrogram{"_tides" if plot_tides else ""}.eps') # , bbox_inches='tight'
    if save_spec:
        np.savetxt(f'/data/localraid/Fibre Data - HW/saved_specs/{window_length}s_window/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}_{freqmax}_{cha1}_spec.txt', spec, delimiter=",")


def ppsd(dir_path:str, prepro_para:dict, t_start:datetime, save_ppsd=False, data=None, window_length=3600):
    cha1, sps, f1, f2, n_minute = prepro_para.get('cha1'), prepro_para.get('samp_freq'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    out_dir = f"./results/figures/PSD_Experiments/{window_length}s_window/ppsds/"
    
    if type(data)==type(None):
        filepath_array, timestamps = get_filepath_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(filepath_array, prepro_para, t_start, duration=timedelta(minutes=n_minute))[:, 0]

    nfft = 2 ** 17
    nr = 201        # number of amplitude bins, changed from 501
    hn = nfft // 2
    seg_length = int(window_length * sps)
    n_segs = len(data) // seg_length
    
    # # Precheck of PSD values to determine range     COMMENTED WHEN SPECIFYING LIMITS BELOW
    # sample_fd_values = []
    # sample_segs =  n_segs // 10 # Sample first 10% of segments
    
    # for i in range(sample_segs):
    #     d = data[i*seg_length:(i+1)*seg_length]
    #     if len(d) == seg_length:
    #         fft_d = np.fft.fft(d, nfft)
    #         psd_lin = (np.abs(fft_d) ** 2) / (nfft * sps)
    #         fd = 10 * np.log10(psd_lin + 1e-12)
    #         sample_fd_values.extend(fd[:hn].flatten())
    # fd_min = np.nanpercentile(sample_fd_values, 0.01)
    # fd_max = np.nanpercentile(sample_fd_values, 99.99)

    fd_min, fd_max = [-40, 50]
    # print(f'fd_min: {fd_min}; fd_max: {fd_max}')
    
    # Create adaptive binning
    db_range = fd_max - fd_min
    scale = (nr - 1) / db_range
    offset = -fd_min
    
    psd = np.zeros((nr, hn))
    p = np.zeros(hn)
    
    f_full = np.arange(nfft) * sps / float(nfft)
    f_pos = f_full[:hn]
    fn1 = np.searchsorted(f_pos, f1, side='left')
    fn2 = np.searchsorted(f_pos, f2, side='right') - 1
    fn1 = np.clip(fn1, 0, hn - 1); fn2 = np.clip(fn2, 0, hn - 1)
    
    for i in range(n_segs):
        d = data[i*seg_length:(i+1)*seg_length]
        ### current fix for NaNs is to skip any section containing NaNs, not exactly robust >:|
        if len(d) == seg_length and ~np.isnan(d).any():
            fft_d = np.fft.fft(d, nfft)
            psd_lin = (np.abs(fft_d) ** 2) / (nfft * sps)
            fd = 10 * np.log10(psd_lin + 1e-12)
            # fd_sm = octave_smooth(fd[:hn], f[:hn])     # octave smoothing
            fd_sm = octave_smooth_gpu(fd[:hn], f_pos)     # octave smoothing gpu
            p += fd_sm
            for j in range(hn):
                index = int((fd_sm[j] + offset) * scale)
                if index < 0:
                    index = 0
                elif index >= nr:
                    index = nr - 1
                psd[index, j] += 1
    
    # Create proper dB axis
    db = np.linspace(fd_min, fd_max, nr)
    # print(f'down: {down_clip}; up: {up_clip}')
    
    nrf = 10
    f = f_pos[fn1:fn2+1]
    f = f[::nrf]
    P = psd[:, fn1: fn2+1]
    pp = savgol_filter(p, 11, 2)
    pp = pp[fn1: fn2+1] / n_segs; pp = pp[::nrf]
    P = P[::1, ::nrf]
    sl = 2
    P = convolve2d(P, np.ones((sl, sl))/sl**2, 'same')
    for i in range(P.shape[1]):
        s = np.sum(P[:, i])
        if s > 0.0:
            P[:, i] /= s
        else:
            P[:, i] = 0.0
    # print(f'PPSD peak percentage after smoothing+renorm: {P.max()*100.0:.2f}%')

    plt.figure(figsize=(6.4,4.8))
    plt.pcolormesh(f, db, P*100, cmap='viridis', vmax=12.0, rasterized=True)
    plt.xscale('log')
    plt.grid(which='both')
    plt.colorbar(aspect=30, pad=0.05, label='Probability [%]', extend='neither')
    plt.semilogx(f, pp, lw=1.2, color='#888888')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Amplitude [$n\epsilon^2/Hz$] [dB]')
    plt.xlim([0.01, 50])
    plt.title(rf"DAS Channel {cha1}")
    # plt.tight_layout()
    plt.subplots_adjust(top=0.922, bottom=0.131, left=0.141, right=0.977, hspace=0.2, wspace=0.2)
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{f1}_{f2}_{cha1}_PPSD.eps') # , bbox_inches='tight'


def get_spec_files(dir:str, t0:datetime, t1:datetime, target_cha:int):
    files = []
    for dir_path, dir_names, file_names in os.walk(dir):
        for file in file_names:
            try:
                if not file.endswith('.npy'): continue
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


def spectral_power_ts(f_ranges:list, window_length:timedelta, target_cha=None, t0=None, t1=None):
    files = get_spec_files('/data/localraid/Fibre Data - HW/saved_specs/600s_window/', t0, t1, target_cha)
    
    mean_df = pd.DataFrame(columns=['timestamp', *[f'{f_range[0]}_{f_range[1]}' for f_range in f_ranges]]).set_index('timestamp')
    t_size = np.load(files[0][0], delimiter=',').shape[1]
    t_bin_length = (files[1][1] - files[0][1]).total_seconds() / t_size
    cosine_window = np.cos(np.linspace(-np.pi/2, np.pi/2, int(window_length.total_seconds() / t_bin_length)))
    cosine_window = cosine_window / cosine_window.sum()
    for file_path, t_start in files:
        # spec = np.loadtxt(file_path, delimiter=',')
        spec = np.load(file_path, mmap_mode="r")
        f_size, t_size = spec.shape
        f_bins = np.linspace(0.0, 50.016666666666666, f_size)
        for f_min, f_max in f_ranges:
            f_idxs = [i for i, f in enumerate(f_bins) if f_min <= f <= f_max]
            freq_mean = spec[f_idxs, :].mean(axis=0)
            filtered = np.convolve(freq_mean, cosine_window, mode='same')
            for j in range(t_size):
                t = t_start + timedelta(seconds=t_bin_length * j)
                mean_df.loc[t, f'{f_min}_{f_max}'] = filtered[j]
    
    # mean_df = pd.DataFrame(mean_fs)
    mean_df.sort_index(inplace=True)
    ax = mean_df.plot(figsize=(10,3))
    
    # storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#', parse_dates=True)
    # for storm in storms_data.index:
    #     dates = storms_data.loc[storm, ['start_date', 'end_date']]
    #     ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
    
    ax.set_ylabel(r'Amplitude [$n\epsilon^2/Hz$] [dB]')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlim(t0, t1)
    ax.grid(axis='x')
    plt.tight_layout()
    f_label = 'high' if f_ranges[0][0] > 1 else 'low'
    plt.savefig(f'./results/figures/ts_spec_lines/{t0}_{t1}_{target_cha}_{f_label}_{window_length.total_seconds()/60}min_avg.eps', bbox_inches='tight')
    # plt.show()


def plot_spectrogram(target_cha, t0, t1, window_len=3600, norm=False, tides=False, markers=None, weather=[], c_range=None, octave_smoothing=True):
    delta = (t1 - t0).total_seconds()
    files = get_spec_files(f'/data/localraid/Fibre Data - HW/saved_specs/{window_len}s_window/', t0, t1, target_cha)
    
    if norm:
        means_df = pd.read_csv(f'./results/checkpoints/{window_len}s_monthly_means_{target_cha}.csv', index_col=0, header=0)
        if delta <= 2419200: means_df = means_df[pd.date_range(t0, t1, freq='MS').strftime("%Y-%m-%d").to_list()]
        else:                means_df = means_df[pd.date_range(t0, t1, freq='MS')[:-1].strftime("%Y-%m-%d").to_list()]
        means = np.asarray(means_df.mean(axis=1))
    ncols = int(np.ceil(len(files)))
    nrows = 1
    if len(weather) > 0:
        nrows += len(weather)
        plot_daily = False if delta <= 2419200 else True
        df_wave = plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily, get_df=True)
        df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily, get_df=True)
        df_era5 = plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily, get_df=True)
        weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[t0.date():t1.date()]
    height_ratios = [2] + [1] * max(0, len(weather))
    if    delta <= 2419200: fig, axs = plt.subplots(nrows, 1, figsize=(12, 4*nrows), sharex=True, gridspec_kw={'height_ratios': height_ratios}, layout='constrained')
    elif len(weather) == 0: fig, axs = plt.subplots(nrows, 1, figsize=(ncols, 8), sharex=True)
    else:                   fig, axs = plt.subplots(nrows, 1, figsize=(ncols, 4*nrows), sharex=True, gridspec_kw={'height_ratios': height_ratios}, layout='constrained')
    
    time_bin_counts = []; spec_min = 0; spec_max = 0
    for f in files:     # there has to be a more efficient way to do this
        # spec = np.loadtext(f[0], delimiter=',')
        spec = np.load(f[0])
        time_bin_counts.append(spec.shape[1])
        if norm: spec -= means[:,np.newaxis]
        spec_min = min(np.nanmin(np.nanpercentile(spec, 1)), spec_min)
        spec_max = max(np.nanmax(np.nanpercentile(spec,99)), spec_max)
        del spec; gc.collect()
    if c_range:
        # if spec_min < c_range[0] or spec_max > c_range[1]:
        #     print(f'spec exceeds given c_range! c_range: {c_range} || spec_min: {spec_min}; spec_max: {spec_max}')        # 15/01/26: commented out while doing normalised plot
        spec_min, spec_max = c_range
    cumulative_offsets = np.cumsum([0] + time_bin_counts[:-1])
    with tqdm(total=len(files), desc=f'Loading and plotting spectrogram files') as pbar:
        for i, (file_path, t_start) in enumerate(files): 
            ax = axs if len(weather) == 0 else axs[0]
            # spec = np.loadtxt(file_path, delimiter=',')
            spec = np.load(file_path)
            if norm: spec -= means[:,np.newaxis]
            
            f_size, t_size = spec.shape
            f_bins = np.linspace(0.01, 50.016666666666666, f_size)
            # if octave_smoothing: spec = octave_smooth(spec, f_bins)
            if octave_smoothing: spec = octave_smooth_gpu(spec, f_bins)
            t_bin_length = (datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S') - t_start).total_seconds() / t_size
            time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]
            col_offset = cumulative_offsets[i]
            y_edges = np.linspace(f_bins[0], f_bins[-1], f_size+1)
            x_edges = np.arange(col_offset, col_offset + t_size + 1)
            # ax.pcolormesh(x_edges, y_edges, spec, cmap='bwr' if norm else 'jet',
            #             vmin=spec_min, vmax=spec_max, shading='auto', rasterized=True)
            ax.imshow(spec, origin='lower', aspect='auto', cmap='bwr' if norm else 'jet', 
                      vmin=spec_min, vmax=spec_max, extent = [col_offset, col_offset + t_size, f_bins[0], f_bins[-1]], rasterized=True)
            ax.set_yscale('log')
            if not hasattr(ax, 'all_time_axes'):
                ax.all_time_axes = []
            ax.all_time_axes.append((col_offset, time_axis))
            del spec; gc.collect()
            pbar.update(1)
    
    if tides or len(weather) > 0:
        spec_times = []
        spec_positions = []
        for col_offset, time_axis in getattr(ax, 'all_time_axes', []):
            for idx, t in enumerate(time_axis):
                spec_times.append(pd.Timestamp(t))
                spec_positions.append(col_offset + idx)
        spec_times = pd.Series(spec_positions, index=spec_times)
        
        if tides:
            ax1 = ax.twinx()
            tidal_df = pd.read_csv('./results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
            tidal_df = tidal_df[t0:t1]

            tidal_x = []
            for ts in tidal_df.index:
                ts = pd.Timestamp(ts)
                diffs = np.abs(spec_times.index - ts)
                if len(diffs) == 0:
                    tidal_x.append(np.nan)
                else:
                    nearest_idx = diffs.argmin()
                    tidal_x.append(spec_times.iloc[nearest_idx])
            tidal_y = tidal_df['ASLVBG02']
            valid = ~np.isnan(tidal_x)
            ax1.plot(np.array(tidal_x)[valid], tidal_y[valid], color='k', alpha=0.5)
            ax1.set_ylabel('Tidal Height (m)')
            plt.sca(ax)
        if len(weather) > 0:
            for weather_ax, (col, label) in zip(axs[1:], weather):
                weather_x = []
                for ts in weather_df.index:
                    ts = pd.Timestamp(ts)
                    diffs = np.abs(spec_times.index - ts)
                    if len(diffs) == 0:
                        weather_x.append(np.nan)
                    else:
                        nearest_idx = diffs.argmin()
                        weather_x.append(spec_times.iloc[nearest_idx])
                valid = ~np.isnan(weather_x)
                weather_ax.plot(np.array(weather_x)[valid], weather_df[col][valid])
                weather_ax.set_ylabel(label)
            plt.sca(ax)
    
    day_starts = []
    for col_offset, time_axis in getattr(ax, 'all_time_axes', []):
        day_starts.extend([(col_offset + idx, t) for idx, t in enumerate(time_axis)])
    
    positions, times = zip(*day_starts)
    times = pd.Series(times, index=positions)

    day_starts = times[times.dt.hour == 0]
    minor_tick_positions = day_starts.index
    
    if markers: 
        marker_positions = []
        for m in markers:
            diffs = np.abs(day_starts.values - np.datetime64(m))
            if len(diffs) > 0:
                idx = diffs.argmin()
                marker_positions.append(day_starts.index[idx])
        ax.plot(marker_positions, [45]*len(marker_positions), 'kv')

    months = pd.Series(times.dt.to_period('M').unique())
    major_tick_positions = []
    major_tick_labels = []
    for month in months:
        month_days = day_starts[day_starts.dt.to_period('M') == month]
        if len(month_days) > 0:
            idxs = np.linspace(0, len(month_days)-1, 3, dtype=int)[:-1]
            for i in idxs:
                major_tick_positions.append(month_days.index[i])
                major_tick_labels.append(month_days.iloc[i].strftime('%Y-%m-%d'))

    ax.set_xticks(major_tick_positions, major_tick_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.tick_params(axis='x', which='minor', length=3, labelsize=0)
    ax.set_ylabel('Frequency [Hz]')
    
    if len(weather) < 0:
        # axs[0].xaxis.set_ticks_position('both')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', which='major', top=True, labeltop=True, bottom=True, labelbottom=True)
        ax.set_xticks(major_tick_positions)
        ax.set_xticklabels(major_tick_labels, rotation=30)
    
    # fig.colorbar(im, ax=axs[-1] if len(weather) > 0 else ax, label='Nano strainrate PSD (dB)', pad=0.40 if len(weather) > 0 else 0.2, aspect=40, orientation="horizontal")
    sm = ScalarMappable(norm=Normalize(vmin=spec_min, vmax=spec_max), cmap=('bwr' if norm else 'jet'))
    sm.set_array([])
    fig.colorbar(sm, ax=axs[-1] if len(weather) > 0 else ax,
                 label=r'Amplitude [$n\epsilon^2/Hz$] [dB]') # , pad=0.40 if len(weather) > 0 else 0.2, aspect=40) # , orientation="horizontal"
    ax.set_title(rf"{t0.date()} to {t1.date()} at channel {target_cha}")
    plt.grid(ax, which='both' if delta <= 2419200 else 'major', linewidth=0.1, alpha=0.5)
    plt.subplots_adjust(hspace=0)
    plt.xticks(rotation=90)
    if len(weather) > 0:
        axs[1].grid(which='major')
        axs[2].grid(which='major')
        plt.sca(axs[-1])
        plt.xticks(rotation=90)
    # else: 
    #     plt.tight_layout()
    del day_starts, months, month_days; gc.collect()
    f_name = f'./results/figures/adapted_specs/spec_{t0}_{t1}_{target_cha}_{window_len}s_{"_tides" if tides else ""}{"_norm" if norm else ""}{"_smooth" if octave_smoothing else ""}.pdf'
    plt.savefig(f_name, bbox_inches='tight')
    plt.close(fig); gc.collect()


def plot_weather_spec(target_cha, t0, t1, window_len=3600, c_range=None, octave_smoothing=True):
    delta = (t1 - t0).total_seconds()
    files = get_spec_files(f'/data/localraid/Fibre Data - HW/saved_specs/{window_len}s_window/', t0, t1, target_cha)
    
    means_df = pd.read_csv(f'./results/checkpoints/{window_len}s_monthly_means_{target_cha}.csv', index_col=0, header=0)
    if delta <= 2419200: means_df = means_df[pd.date_range(t0, t1, freq='MS').strftime("%Y-%m-%d").to_list()]
    else:                means_df = means_df[pd.date_range(t0, t1, freq='MS')[:-1].strftime("%Y-%m-%d").to_list()]
    means = np.asarray(means_df.mean(axis=1))
    
    plot_daily = False # if delta <= 2419200 else True
    df_wave = plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily, get_df=True)
    df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily, get_df=True)
    df_era5 = plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily, get_df=True)
    weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[t0.date():t1.date()]
    weather_df = weather_df.resample(timedelta(hours=1)).mean()
    
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    
    time_bin_counts = []; spec_min = 0; spec_max = 0
    for f in files:     # there has to be a more efficient way to do this
        # spec = np.loadtxt(f[0], delimiter=',')
        spec = np.load(file_path, mmap_mode="r")
        time_bin_counts.append(spec.shape[1])
        spec -= means[:,np.newaxis]
        spec_min = min(np.nanmin(np.nanpercentile(spec, 1)), spec_min)
        spec_max = max(np.nanmax(np.nanpercentile(spec,99)), spec_max)
        del spec; gc.collect()
    if c_range:
        # if spec_min < c_range[0] or spec_max > c_range[1]:
        #     print(f'spec exceeds given c_range! c_range: {c_range} || spec_min: {spec_min}; spec_max: {spec_max}')        # 15/01/26: commented out while doing normalised plot
        spec_min, spec_max = c_range
    cumulative_offsets = np.cumsum([0] + time_bin_counts[:-1])
    with tqdm(total=len(files), desc=f'Loading and plotting spectrogram files') as pbar:
        for i, (file_path, t_start) in enumerate(files): 
            # spec = np.loadtxt(file_path, delimiter=',')
            spec = np.load(file_path, mmap_mode="r")
            spec -= means[:,np.newaxis]
            
            f_size, t_size = spec.shape
            f_bins = np.linspace(0.01, 50.016666666666666, f_size)
            if octave_smoothing: spec = octave_smooth_gpu(spec, f_bins)
            t_bin_length = (datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S') - t_start).total_seconds() / t_size
            time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]
            col_offset = cumulative_offsets[i]
            y_edges = np.linspace(f_bins[0], f_bins[-1], f_size+1)
            x_edges = np.arange(col_offset, col_offset + t_size + 1)
            ax.pcolormesh(x_edges, y_edges, spec, cmap='bwr',
                        vmin=spec_min, vmax=spec_max, shading='auto', rasterized=True)
            ax.set_yscale('log')
            if not hasattr(ax, 'all_time_axes'):
                ax.all_time_axes = []
            ax.all_time_axes.append((col_offset, time_axis))
            del spec; gc.collect()
            pbar.update(1)
    
    spec_times = []
    spec_positions = []
    for col_offset, time_axis in ax.all_time_axes:
        for idx, t in enumerate(time_axis):
            spec_times.append(pd.Timestamp(t))
            spec_positions.append(col_offset + idx)

    spec_times = pd.Series(spec_positions, index=spec_times)
    weather_x = []
    for ts in weather_df.index:
        ts = pd.Timestamp(ts)
        diffs = np.abs(spec_times.index - ts)
        if len(diffs) == 0:
            weather_x.append(np.nan)
        else:
            nearest_idx = diffs.argmin()
            weather_x.append(spec_times.iloc[nearest_idx])

    for col, label, f_low, f_high in [["Wind(m/s)", 'Wind speed (m/s)', 0.2, 0.5], ['Hs(Hm0)(m)', 'Sig. Wave height (m)', 10.0, 20.0]]:
        weather_x = np.array(weather_x)
        valid = ~np.isnan(weather_x)
        w = weather_df[col].astype(float).to_numpy()
        w_min, w_max = np.nanmin(w), np.nanmax(w)

        def w_to_freq(ws, f0, f1):
            return f_low + (np.clip(ws, w_min, w_max)-w_min)/(w_max-w_min+1e-12)*(f1 - f0)
        
        w_freq = w_to_freq(w, f_low, f_high)
        ax.plot(weather_x[valid], w_freq[valid], color='k',
                lw=1.2, alpha=0.5, zorder=10)

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        y0_ax = ax.transAxes.inverted().transform(ax.transData.transform((0, f_low)))[1]
        y1_ax = ax.transAxes.inverted().transform(ax.transData.transform((0, f_high)))[1]
        band_h = y1_ax - y0_ax

        ax_w = inset_axes(
            ax,
            width="8%",
            height="100%",
            loc="upper right",
            bbox_to_anchor=(0.0, y0_ax, 1.0, band_h),
            bbox_transform=ax.transAxes,
            borderpad=0.0
        )
        ax_w.patch.set_alpha(0)
        ax_w.set_ylim(min(0, w_min), w_max)

        ax_w.yaxis.set_label_position("right")
        ax_w.yaxis.tick_right()
        ax_w.set_ylabel(label, size=8)
        ax_w.spines['top'].set_visible(False)
        ax_w.spines['bottom'].set_visible(False)
        ax_w.spines['left'].set_visible(False)
        ax_w.set_xticks([])    
    
    day_starts = []
    for col_offset, time_axis in getattr(ax, 'all_time_axes', []):
        day_starts.extend([(col_offset + idx, t) for idx, t in enumerate(time_axis)])
    positions, times = zip(*day_starts)
    times = pd.Series(times, index=positions)
    day_starts = times[times.dt.hour == 0]
    minor_tick_positions = day_starts.index
    months = pd.Series(times.dt.to_period('M').unique())
    major_tick_positions = []
    major_tick_labels = []
    for month in months:
        month_days = day_starts[day_starts.dt.to_period('M') == month]
        if len(month_days) > 0:
            idxs = np.linspace(0, len(month_days)-1, 6, dtype=int)[:-1]
            for i in idxs:
                major_tick_positions.append(month_days.index[i])
                major_tick_labels.append(month_days.iloc[i].strftime('%Y-%m-%d'))
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    sm = ScalarMappable(norm=Normalize(vmin=spec_min, vmax=spec_max),
                        cmap='bwr')
    sm.set_array([])
    # use a small fraction + pad so horizontal colorbar is placed outside axes without overlapping
    cbar = fig.colorbar(sm, ax=ax, label=r'Amplitude [$n\epsilon^2/Hz$] [dB]') # , pad=0.07
                        # pad=0.02, fraction=0.035, aspect=40)
    # fig.subplots_adjust(right=0.2)
    ax.set_title(rf"Differential Spectrogram at Channel {target_cha}")
    ax.set_xticks(major_tick_positions, major_tick_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.tick_params(axis='x', which='minor', length=2, labelsize=0)
    
    ax.tick_params(axis='x', which='both',
                bottom=True, top=False, labelbottom=True)  # rotate x tick labels
    ax.tick_params(axis='y', which='both',
                left=True, right=False, labelleft=True, labelright=False)
    ax.set_ylabel('Frequency [Hz]')
    # plt.setp(ax.get_xticklabels(), rotation=90)
    # plt.xticks(rotation=90)
    # plt.grid(ax, which='both' if delta <= 2419200 else 'major', linewidth=0.1, alpha=0.5)
    # del day_starts, months, month_days; gc.collect()
    f_name = f'./results/figures/adapted_specs/months_norm/spec_{t0}_{t1}_{target_cha}_{window_len}s_{"_smooth" if octave_smoothing else ""}.eps'
    plt.subplots_adjust(top=0.925, bottom=0.154, left=0.091, right=1.0, hspace=0.2, wspace=0.2)
    plt.savefig(f_name)
    plt.show()
    plt.close(fig); gc.collect()


def get_spectral_mean(target_cha=None, t0=None, t1=None, window_len=3600):
    files = get_spec_files(f'/data/localraid/Fibre Data - HW/saved_specs/{window_len}s_window/', t0, t1, target_cha)
    mean_df = pd.DataFrame()
    for file_path, t_start in files:
        # spec = np.loadtxt(file_path, delimiter=',')
        spec = np.load(file_path, mmap_mode="r")
        means = np.nanmean(spec, axis=1)
        mean_df = pd.concat([mean_df, pd.Series([t_start.date(), *means])], axis=1)
    mean_df.to_csv(f'./results/checkpoints/{window_len}s_monthly_means_{target_cha}.csv', header=False)


def calc_angle_between_points(lat1, lon1, lat2, lon2):
    '''Input lat-lons as degrees!!!'''
    lat1, lon1, lat2, lon2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
    
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    angle = degrees(atan2(y, x))
    angle = (angle + 360) % 360
    return angle


def sensitivity_analysis(gps_track:pd.DataFrame, target_ch, plot=True):
    angles = []
    for current_ch in gps_track.index:
        if current_ch == gps_track.index[0]:
            current_ch += 1
        elif current_ch == gps_track.index[-1]:
            current_ch -= 1
        lat1, lon1 = gps_track.loc[current_ch - 1, ['lat', 'lon']]
        lat2, lon2 = gps_track.loc[current_ch + 1, ['lat', 'lon']]
        angles.append(calc_angle_between_points(lat1, lon1, lat2, lon2))
    
    gps_track['relative_angle'] = angles
    target_angle = gps_track.loc[current_ch, 'relative_angle']
    relative_angles = []
    long_sens = []
    trans_sens = []
    for current_ch in gps_track.index:
        angle = gps_track.loc[current_ch, 'relative_angle'] - target_angle
        relative_angles.append(abs(angle))
        long_sens.append(abs(cos(radians(angle))))
        trans_sens.append(sin(2*radians(angle)) ** 2)
    
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(6.4,4.8))
        for i, arr in enumerate([relative_angles, long_sens, trans_sens]):
            im = axs[i].scatter(gps_track['lon'], gps_track['lat'], c=arr, cmap='seismic')
            fig.colorbar(im, ax=axs[i])
            axs[i].scatter(gps_track.loc[target_ch, 'lon'], gps_track.loc[target_ch, 'lat'], s=100, c='k')
        plt.tight_layout()
        plt.show()
    
    return np.sum(long_sens), np.sum(trans_sens)


def band_amp_bulk_runner():
    dir_path = '/data/QNAP1_Data/Data/'
    
    f_bands = [[0.01, 0.1], [0.1, 0.3], [0.3, 0.8], [1.0, 5.0], [5.0, 10.0]]
    for y, m in [[2024, 11], [2024, 12], [2025, 1], [2025, 2], [2025, 3], [2025, 4], [2025, 5], [2025, 6]]:
        print(f'Running {y}/{m}')
        t_start = datetime(year=y, month=m, day=1)
        t_end = t_start + relativedelta(months=1)
        n_minute = (t_end - t_start).total_seconds() // 60
        prepro_para = {
            'samp_freq': 100,
            'target_spatial_res': 1,
            'n_minute': n_minute,
            'freqmin': 0.01,
            'freqmax': 25.0,
            'cha1': 0,
            'cha2': None,
        }
        
        get_band_amplitudes(dir_path, prepro_para, t_start, f_bands)


def get_band_amplitudes(dir_path, prepro_para, t_start, f_bands, chunk_minutes=60):
    n_minute = prepro_para.get("n_minute")
    t_end = t_start + timedelta(minutes=n_minute)

    filepath_array, timestamps = get_filepath_array(dir_path, t_start, t_end)

    tdms = TdmsReader(filepath_array[0])
    props = tdms.get_properties()
    n_chas = tdms.fileinfo["n_channels"]
    prepro_para.update({"cha2": n_chas - 1})

    running_sum = [None] * len(f_bands)
    running_count = np.zeros(len(f_bands))

    current_time = t_start
    while current_time < t_end:
        data = get_data_from_array(
            filepath_array,
            prepro_para,
            current_time,
            duration=timedelta(minutes=chunk_minutes),
            filter=False,
        )

        valid_rows = ~np.isnan(data).any(axis=1)
        data = data[valid_rows]

        freqs, psd = welch(
            data,
            fs=prepro_para["samp_freq"],
            axis=0,
            nperseg=4096,
        )

        for i, (fmin, fmax) in enumerate(f_bands):

            mask = (freqs >= fmin) & (freqs <= fmax)

            if len(mask.shape) == 1:
                band_power = np.sum(psd[mask, :], axis=0)
            else:
                print(f'broke for {fmin} at {current_time}')
                continue

            if running_sum[i] is None:
                running_sum[i] = np.zeros_like(band_power)

            running_sum[i] += band_power
            running_count[i] += mask.sum()

        current_time += timedelta(minutes=chunk_minutes)

    cols = [f'{fmin}_{fmax}' for (fmin, fmax) in f_bands]
    results_df = pd.DataFrame(index=range(0,n_chas), columns=cols)
    for i, (fmin, fmax) in enumerate(f_bands):

        mean_psd = running_sum[i] / running_count[i]
        log_psd = 10 * np.log10(mean_psd)

        results_df[f'{fmin}_{fmax}'] = log_psd

        np.savetxt(f"./results/cha_spectra/{t_start.date()}_{t_end.date()}_{fmin}_{fmax}.txt", log_psd)

    print(results_df)
    results_df.to_csv(f"./results/cha_spectra/{t_start.date()}_{t_end.date()}.txt")


def map_amplitude_plots(band_amplitudes, gps_track:pd.DataFrame, cmap='viridis', basemap=True):    
    os_request = cimgt.StadiaMapsTiles(apikey='c3e0a719-1e2c-4797-9276-16a7292f91a8', style='stamen_terrain')
    fig, ax = plt.subplots(figsize=(7,6), subplot_kw={'projection': os_request.crs})
    
    # clip exposed channels
    gps_track = gps_track[~gps_track['channel_no'].between(1035, 1100)]
    gps_track = gps_track[~gps_track['channel_no'].between(1515, 1570)]
    
    band_amplitudes = band_amplitudes[(band_amplitudes.index < 1035) | (band_amplitudes.index > 1100)]
    band_amplitudes = band_amplitudes[(band_amplitudes.index < 1515) | (band_amplitudes.index > 1570)]
    
    all_values = []
    for band in band_amplitudes:
        amps = band_amplitudes[band]
        vals = amps.dropna().values
        all_values.extend(vals)
    
    norm = Normalize(vmin=np.nanpercentile(all_values, 1), vmax=np.nanpercentile(all_values, 99))

    gps_track_cp = gps_track.copy()
    for i, band in enumerate(band_amplitudes):
        gps_track = gps_track_cp.copy()
        
        amps = band_amplitudes[band]
        gps_track["amplitude"] = gps_track["channel_no"].map(amps)

        # ensure channel ordering
        gps_track = gps_track.sort_values("channel_no").reset_index(drop=True)
        gps_track = gps_track[~np.isnan(gps_track).any(axis=1)]
        
        gps_track['lat'] = gps_track['lat'] - (i*0.002)

        # project to web mercator
        gdf = gpd.GeoDataFrame(
            gps_track,
            geometry=gpd.points_from_xy(gps_track["lon"], gps_track["lat"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        segments = []
        values = []

        for i in range(len(gdf) - 1):
            p1 = gdf.geometry.iloc[i]
            p2 = gdf.geometry.iloc[i + 1]

            segments.append([(p1.x, p1.y), (p2.x, p2.y)])
            values.append(np.nanmean([gdf["amplitude"].iloc[i], gdf["amplitude"].iloc[i + 1]]))
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4)

        lc.set_array(np.asarray(values))
        ax.add_collection(lc)
        
        band = band.split('_')
        ax.text(p2.x-300, p2.y, f'{band[0]}-{band[1]} Hz')

    # ax.autoscale()
    if basemap:
        # cx.add_basemap(ax, crs=gdf.crs)
        ext = [1.365, 1.390, 52.890, 52.907]
        ax.set_extent(ext)
        ax.add_image(os_request, 15)
    
    # gridlines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    # gridlines.xformatter = LONGITUDE_FORMATTER
    # gridlines.yformatter = LATITUDE_FORMATTER
    # gridlines.top_labels = False
    # gridlines.right_labels = False

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Mean Amplitude [$n\epsilon^2/Hz$] [dB]")
    # ax.set_title("Cable spectral amplitude")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':  
    # parallel_spectral_analysis()
    
    ### cable amplitude channel maps
    # band_amp_bulk_runner()
    # m = 4
    # band_amplitudes = pd.read_csv(f'./results/cha_spectra/2025-0{m}-01_2025-0{m+1}-01.txt', index_col=0)
    # gps_track = pd.read_csv('./results/checkpoints/final_gps_track_1m.csv')
    # map_amplitude_plots(band_amplitudes, gps_track)
    
    ### all the weather stuff
    markers_arr = [
        # datetime(2024, 5, 30),
        datetime(2024, 4, 6),       # Kathleen
        datetime(2024, 8, 22),      # Lilian
        # datetime(2024, 9, 27),
        # datetime(2024, 10, 10),
        datetime(2024, 10, 20),     # Ashley
        datetime(2024, 11, 22),     # Bert
        datetime(2024, 11, 27),     # Conall
        datetime(2024, 12, 6),      # Darragh
        datetime(2025, 1, 24),      # Eowyn
        # datetime(2025, 2, 7),
        # datetime(2025, 3, 11),
        # datetime(2025, 4, 2),
    ]
    
    # t0 = datetime(year=2024, month=4, day=1); t1 = datetime(year=2025, month=7, day=1)
    # for cha in [750, 788, 875, 1475]:
    #     get_spectral_mean(cha, t0, t1)
    
    weather_arr = [['Wind(m/s)','Wind speed (m/s)'],['Hs(Hm0)(m)','Wave height (m)']]
    # avg_time = timedelta(days=5)
    c_range_arr = [-30.128117506053417, 38.67278348993597]              # for all non-norm specs 2024-04 to 2025-07
    c_range_norm_arr = [-27.121361009693757, 26.618877627443347]        # for norm specs
    
    c_range_nodes_arr = [-32.7434695431312, 33.040064418459345]
    c_range_norm_nodes_arr = [-23.176340627918208, 24.762160092331]
    
    t0 = datetime(year=2024, month=4, day=1); t1 = datetime(year=2025, month=4, day=1)
    window_len = 3600
    with tqdm(total=2, desc=f"Spectrograms - Aprils", position=0) as pbar:
        for cha in [1475]:  # 750, 788, 875, 
            plot_spectrogram(target_cha=cha, t0=t0, t1=t1, window_len=window_len); pbar.update(1)
            plot_spectrogram(target_cha=cha, t0=t0, t1=t1, window_len=window_len, norm=True, markers=markers_arr, weather=weather_arr, c_range=c_range_norm_arr); pbar.update(1)

    # for y, m in [[2024, 12], [2025, 5]]:
    #     t0 = datetime(year=y, month=m, day=1)
    #     t1 = t0 + relativedelta(months=1)
    #     with tqdm(total=8, desc=f"Spectrograms - {y}/{m}", position=0) as pbar:
    #         for cha in [750, 1475]:
    #             plot_weather_spec(cha, t0, t1, window_len=600, c_range=c_range_nodes_arr); pbar.update(1)
    #             plot_weather_spec(cha, t0, t1, window_len=3600, c_range=c_range_nodes_arr); pbar.update(1)
    