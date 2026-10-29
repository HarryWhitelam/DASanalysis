import gc
from TDMS_Read import TdmsReader
from SegyReader import SegyReader
import os
import warnings
import numpy as np
from tqdm import tqdm
from obspy.signal.filter import bandpass
from datetime import datetime, timedelta
from math import ceil
from scipy.signal import decimate


def get_reader_array(root_dir:str, t_start:datetime=None, t_end:datetime=None):
    files = []
    for dir_path, dir_names, file_names in os.walk(root_dir):
        for file in file_names:
            if file.endswith(('.tdms', '.segy')): 
                try: 
                    timestamp = datetime.strptime(file[-24:-9], '%Y%m%d_%H%M%S')
                except Exception: 
                    continue        # basically there was no datetime within the file
                if t_start and timestamp < t_start:
                    continue
                if t_end and timestamp > t_end:
                    continue
                files.append((TdmsReader(os.path.join(dir_path, file)), timestamp))
    files.sort(key=lambda x: x[1])
    reader_array, timestamps = zip(*files)
    return list(reader_array), np.asarray(timestamps, dtype=datetime)


def get_filepath_array(root_dir:str, t_start:datetime=None, t_end:datetime=None):
    files = []
    # exclude = ['20250326_Jack_Experiments', 'Archive']
    exclude = ['Archive']
    for dir_path, dir_names, file_names in os.walk(root_dir, topdown=True):
        dir_names[:] = [d for d in dir_names if d not in exclude]
        for file in file_names:
            if file.endswith(('.tdms', '.segy')): 
                try: 
                    timestamp = datetime.strptime(file[-24:-9], '%Y%m%d_%H%M%S')
                except Exception: 
                    continue        # basically there was no datetime within the file
                if t_start and timestamp < t_start:
                    continue
                if t_end and timestamp > t_end:
                    continue
                files.append((os.path.join(dir_path, file), timestamp))
    if len(files) == 0:
        warnings.warn(f'No files found between {t_start} and {t_end}!')
        return 0
    files.sort(key=lambda x: x[1])
    reader_array, timestamps = zip(*files)
    return list(reader_array), np.asarray(timestamps, dtype=datetime)


def get_closest_index_before(timestamps:np.ndarray, time:datetime, inclusive:bool=True):
    """
    Return the closest timestamp in `timestamps` that is before (or equal to, if inclusive=True) `time`.
    Returns None if no such timestamp exists.

    Args:
        timestamps: 1D sorted numpy array of datetimes
        time: target datetime
        inclusive: if True, allow a timestamp equal to `time`; if False, require strictly earlier

    Returns:
        datetime or None
    """
    side = 'right' if inclusive else 'left'
    idx = timestamps.searchsorted(time, side=side)
    if idx == 0:
        return None
    return idx - 1


def get_dir_properties(dir_path:str):
    # with os.scandir(dir_path) as files:
    for dir_path, dir_names, file_names in os.walk(dir_path, topdown=True):
        for file in file_names:
            if file.endswith(('.tdms', '.segy')):
                match file.rsplit('.', 1)[-1]:
                    case 'tdms': file_reader = TdmsReader(f'{dir_path}/{file}')
                    case 'segy': file_reader = SegyReader(f'{dir_path}/{file}')
                    case _: print("BAD FILE EXTENSION!!!!!!")
                break
    file_reader._read_properties()
    return file_reader.get_properties()


# adapting this so that all channel requests work off of 1m channels - as though distance along cable
def get_data_from_array(data_array:list, prepro_para:dict, start_time:datetime, duration:timedelta, channels=None, filter=True):
    cha1                = prepro_para.get('cha1')
    cha2                = prepro_para.get('cha2')
    samp_freq           = prepro_para.get('samp_freq')
    target_spatial_res  = prepro_para.get('target_spatial_res')
    freqmin             = prepro_para.get('freqmin')
    freqmax             = prepro_para.get('freqmax')
    rcv_ch              = prepro_para.get('rcv_ch')
    
    if rcv_ch:
        channels = [cha1, rcv_ch]
        
    if cha1 == None: 
        cha1 = 0
    if cha2 == None:
        if type(data_array[0]) == str:
            file = TdmsReader(data_array[0])
        else:
            file = data_array[0]
        props = file.get_properties()
        cha2 = (file.fileinfo['n_channels'] / int(target_spatial_res/props.get('SpatialResolution[m]'))) - 1
    
    current_time = 0
    if channels: 
        tdata = np.empty((int(duration.total_seconds() * samp_freq), len(channels)))
    else: 
        tdata = np.empty((int(duration.total_seconds() * samp_freq), ceil(((cha2-cha1)/target_spatial_res)+1)))     # 20260723: changed from (cha2-cha1+1)/target_spatial_res
    tdata.fill(np.NaN)
    
    with tqdm(total=(duration.total_seconds()), desc=f'[Process {os.getpid()}] Loading data (s)', position=1, leave=False) as pbar:
        while current_time < duration.total_seconds() and len(data_array) != 0:
            data_file = data_array.pop(0)
            if type(data_file) == str:
                try:
                    data_file = TdmsReader(data_file)
                except: 
                    print(f'Failed at {data_file}')
            props = data_file.get_properties()
            # print(data_file.fileinfo['n_channels'])
            
            props_ts = props.get('GPSTimeStamp').replace(microsecond=0)
            expected_ts = start_time + timedelta(seconds=current_time)
            delta_secs = int(round((props_ts - expected_ts).total_seconds()))
            if delta_secs > 0:
                pbar.update(delta_secs)
                current_time += delta_secs
                expected_ts = start_time + timedelta(seconds=current_time)
            skip_seconds = 0
            if delta_secs < 0:
                skip_seconds = -delta_secs
            
            spatial_ratio = int(target_spatial_res/props.get('SpatialResolution[m]'))
            if channels:
                file_channels = [int(channel/props.get('SpatialResolution[m]')) for channel in channels]
                try:
                    data = data_file.get_data(file_channels[0], file_channels[-1])[:,np.array(file_channels)-file_channels[0]]
                except:
                    print(f"Failed at {props_ts}! File_channels: {file_channels}")
            else:
                c1, c2 = int(cha1/props.get('SpatialResolution[m]')), int(cha2/props.get('SpatialResolution[m]'))   # get channels for each file
                data = data_file.get_data(c1, c2)
                data = data[:, ::spatial_ratio]
            data = scale(data, props)
            if props.get('SamplingFrequency[Hz]') > samp_freq:
                data = decimate(data,
                                int(props.get('SamplingFrequency[Hz]') / samp_freq),
                                ftype='iir',
                                zero_phase=True, 
                                axis=0)
            elif props.get('SamplingFrequency[Hz]') < samp_freq:
                warnings.warn(f"Sampling frequency below target frequency! Timestamp: {props.get('GPSTimeStamp')}; fs: {props.get('SamplingFrequency[Hz]')}")
            if data.shape[1] == 0:
                print(props.get('GPSTimeStamp'))
                print(f'SHAPE: {data.shape}')
                print(f'Channels: {c1}, {c2} || {cha1}, {cha2}')
                print(f'SpatialRes: {props.get("SpatialResolution[m]")}')
            if filter: 
                data = np.float32(bandpass(data,
                                           0.9 * freqmin,
                                           1.1 * freqmax,
                                           df=samp_freq,
                                           corners=4,
                                           zerophase=True))
            if 'skip_seconds' in locals() and skip_seconds > 0:
                skip_rows = int(skip_seconds * samp_freq)
                if skip_rows >= data.shape[0]:
                    # this file finishes before the expected start time -> skip it
                    pbar.update(data.shape[0] / float(samp_freq))
                    del data
                    continue
                data = data[skip_rows:, :]
            current_row = int(current_time * samp_freq)
            t_size = data.shape[0]
            if t_size > tdata.shape[0] - current_row:        # clips final file
                data = data[:tdata.shape[0]-current_row,:]
                t_size = data.shape[0]
            try:
                tdata[current_row:current_row+t_size, :] = data
            except: 
                print(f"Loading failed at {start_time}, current_time: {current_time}.")
                print(data.shape)
                print(tdata.shape)
                print(current_row, t_size)
            current_time += t_size/samp_freq
            
            # attempting to avoid issue of open file limit
            data_file._tdms_file.close()
            del data_file
            gc.collect()
            pbar.update(t_size/samp_freq)
    return tdata


def scale(data:np.ndarray, props:dict):
    """Takes in DAS data and its properties using them to scale the data as it is compressed within the file format. Returns scaled data

    strainrate nm/m/s = 116 * iDAS values * sampling freq (Hz) / gauge lenth (m)

    Keyword arguments:
        data -- numpy array containing DAS data
        props -- properties struct from DAS reader
    """
    data = data / 8192
    data = (116 * data * props.get('SamplingFrequency[Hz]')) / props.get('GaugeLength')
    return data
