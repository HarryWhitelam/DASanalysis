import os
from obspy import read, read_inventory, UTCDateTime
from obspy.signal import PPSD
from datetime import timedelta
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import gc


def rotate_node_data(station, t0):
    st  = read(f'/data/localraid/UEA_nodes_mseed_mV/corrected/5J.{station}..DPN..{t0}.MSEED')
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/corrected/5J.{station}..DPE..{t0}.MSEED')
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/corrected/5J.{station}..DPZ..{t0}.MSEED')
    
    for tr in st:
        if tr.stats.component == 'N':
            tr.stats.component = '1'
        elif tr.stats.component == 'E':
            tr.stats.component = '2'

    inv = read_inventory('./station.xml')
    st.rotate(method='->ZNE', inventory=inv)
    
    for tr in st:
        tr.write(rf"/data/localraid/UEA_nodes_mseed_mV/rotated/{tr.id}..{tr.stats.starttime}.MSEED", format="MSEED")


def prep_node_data(station, t0, save_resp_rem=True):
    asn = 1 if t0.day == 31 else t0.day+1     # asn = annoying specific number
    t_format = t0.strftime('%Y.%m.%d.%H.%M.%S')
    station = '4530' + station
    
    st = read(f'/data/localraid/UEA_nodes_mseed_mV/{station}.{asn}.{t_format}.000.N.miniseed')
    st[0].attach_response(read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station[-5:]}..DPN.76_6'))
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/{station}.{asn}.{t_format}.000.E.miniseed')
    st[1].attach_response(read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station[-5:]}..DPE.76_6'))
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/{station}.{asn}.{t_format}.000.Z.miniseed')
    st[2].attach_response(read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station[-5:]}..DPZ.76_6'))

    for tr in st: tr.data *= 0.001      # convert from mV to V
    st.decimate(16)     # decimate to 250 Hz (factor of 16)
    
    if save_resp_rem:
        t1 = t0 + (60*60*12)
        tr = st[2].slice(t0, t1)
        tr.remove_response(pre_filt=[0.01, 0.05, 99, 100], water_level=30, plot=f'resp_rem/{tr.id}..{tr.stats.starttime}.png')

    for tr in st: tr.remove_response(pre_filt=[0.01, 0.05, 99, 100], water_level=30)
    # st.plot()

    for tr in st:
        tr.write(f"/data/localraid/UEA_nodes_mseed_mV/corrected/{tr.id}..{tr.stats.starttime}.MSEED", format="MSEED")
    
    del st
    del tr
    gc.collect()

# stations = ['04915', '05014', '03479', '03713', '05330', '05100']
# t_start = UTCDateTime('2025-04-01T:00:00.00')
# n_days = 10

# prep_node_data(station, t_start)

# with tqdm(total=len(stations) * n_days) as pbar:
#     for station in stations:
#         for t0 in [t_start + timedelta(days=x) for x in range(n_days)]:
#             prep_node_data(station, t0)
#             rotate_node_data(station, t0)
#             pbar.update(1)
