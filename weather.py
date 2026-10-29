import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from windrose import WindroseAxes
import xarray as xr
from math import pi
from collections import Counter



def dms_to_dd(degrees, minutes=0, seconds=0):
    return degrees + (minutes/60) + (seconds/3600)


def plot_weather():
    weather_data = pd.read_csv('./results/checkpoints/weather.csv', sep=',', index_col=[0, 1], comment='#', na_values=['   --- ', '   ---'])
    weather_data.index = [np.datetime64(f'{date[0]}-{date[1] if date[1] > 9 else f"0{date[1]}"}', 'D') for date in weather_data.index]
    # print(weather_data)
    deployment_data = weather_data.loc[np.datetime64('2023-09-01'):]
    axs = deployment_data.plot.line(None, subplots=True, legend=False, grid=True, figsize=(12, 12))
    for ax, label in zip(axs, ['Max temp (degC)', 'Min temp (degC)', 'AF (days)', 'Rainfall (mm)', 'Sun (hours)']):
        ax.set_ylabel(label)
    plt.tight_layout()
    plt.savefig('./results/figures/weather_data.eps')
    plt.show()


def plot_rain_storms():
    weather_data = pd.read_csv('./results/checkpoints/weather.csv', sep=',', index_col=[0, 1], comment='#', na_values=['   --- ', '   ---'], skipinitialspace=True)
    weather_data.index = [np.datetime64(f'{date[0]}-{date[1] if date[1] > 9 else f"0{date[1]}"}', 'D') for date in weather_data.index]
    rain_data = weather_data.loc[np.datetime64('2023-09-01'):, ['rain']]
    rain_data['rain'] = rain_data['rain'].astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(rain_data.index, rain_data['rain'])
    ax.set_ylabel('Rainfall (mm)')
    ax.grid(which='both') 
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=()))
    
    storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
    prev_storm_end = 0
    for storm in storms_data.index:
        dates = storms_data.loc[storm, ['start_date', 'end_date']]
        ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.5)
        if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
            ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
        else:
            ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
        prev_storm_end = np.datetime64(dates['end_date'])
    plt.tight_layout()
    # plt.savefig('./results/figures/rainfall_storms.eps')
    plt.show()


def plot_era5_data(file_path:str):
    import xarray as xr
    import cartopy.crs as ccrs
    
    if file_path.split('.')[-1 == 'grib']:
        data = xr.open_dataset(file_path, engine='cfgrib')
        print(data)
        times = data.time.values
        steps = data.step.values
        print(times)
        
        df = pd.DataFrame(columns=['timestamp','V','rainfall'])
        for time in times:
            arr = []
            hour_data = data.sel(time=time, longitude=1.50, latitude=53.0)
            print(hour_data)
            V = np.sqrt(np.square(hour_data.v10.values) + np.square(hour_data.u10.values))
            print(time, V)
            for step in steps:
                rain_data = data.sel(time=time, step=step, longitude=1.50, latitude=53.0)
                print(rain_data.tp.values)

            # arr.append([time, V, rainfall])
        df = df.set_index('timestamp')
        print(df)
        df.to_csv(f'./results/checkpoints/hourly_era5.csv')
        
        plot_time = 'daily'
        # df = df.groupby(pd.to_datetime(df.index).date).agg({'rainfall(mm)': 'sum'}).reset_index()
        df = df.groupby(pd.to_datetime(df.index).date).agg({'u10(m/s)': 'mean', 'v10(m/s)': 'mean'})
        df.index.name = 'timestamp'
        print(df)
        df.to_csv(f'./results/checkpoints/daily_{var}.csv')
    else: 
        var = file_path.split('.')[-2].split('_')[-1]
        plot_time = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path, parse_dates=['timestamp'], header=0)
        df = df.set_index('timestamp')        
        if 'windspeed' in file_path:
            df['V'] = np.sqrt(np.square(df['u10(m/s)']) + np.square(df['v10(m/s)']))
            df['angle'] = np.mod(180 + 180/pi * np.arctan2(df['u10(m/s)'],df['v10(m/s)']), 360)
        print(df)
    
    start_date = np.datetime64('2023-09-01')
    
    fig, ax = plt.subplots()
    df = df[start_date:]
    # ax.bar(mdates.date2num(df.index), df['rainfall(mm)'])
    # ax.set_ylabel(f'{plot_time.capitalize()} rainfall (mm)');
    # ax.plot(mdates.date2num(df.index), df['u10(m/s)'], label='u10')
    # ax.plot(mdates.date2num(df.index), df['v10(m/s)'], label='v10')
    ax.plot(mdates.date2num(df.index), df['V'], label='V')
    # ax2 = ax.twinx()
    # ax2.plot(mdates.date2num(df.index), df['angle'], label='angle', color=(0.761, 0.455, 0, 0.5))
    # plt.legend()
    ax.set_ylabel(f'Average {plot_time} wind speed (m/s)')
    # ax2.set_ylabel(f'{plot_time.capitalize()} wind angle')
    # ax2.yaxis.label.set_color((0.761, 0.455, 0))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    
    storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
    prev_storm_end = 0
    for storm in storms_data.index:
        dates = storms_data.loc[storm, ['start_date', 'end_date']]
        ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
        # if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
        #     ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
        # else:
        #     ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
        # prev_storm_end = np.datetime64(dates['end_date'])
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./results/figures/{plot_time}_{var}.eps')
    
    
def era5_data_to_csv(file_path:str):
    try:
        wind_data = xr.open_dataset(file_path, engine='cfgrib', 
                                    filter_by_keys={'shortName': ['10u', '10v']})
        wind_data = wind_data.sel(longitude=1.50, latitude=53.0, method='nearest')
    except Exception as e:
        print(f"Error reading wind data: {e}")
        return
    
    try:
        precip_data = xr.open_dataset(file_path, engine='cfgrib', 
                                    filter_by_keys={'shortName': 'tp'})
        precip_data = precip_data.sel(longitude=1.50, latitude=53.0, method='nearest')
    except Exception as e:
        print(f"Error reading precipitation data: {e}")
        return
    
    df_list = []
    for time in precip_data.time.values:
        for step in precip_data.step.values:
            wind_time = pd.to_datetime(time) + pd.to_timedelta(step)
            try:
                wind_slice = wind_data.sel(time=wind_time)
                u10 = float(wind_slice.u10.values) if 'u10' in wind_slice else 0
                v10 = float(wind_slice.v10.values) if 'v10' in wind_slice else 0
                V = np.sqrt(u10**2 + v10**2)
                V_angle = np.mod(180 + 180/pi * np.arctan2(u10,v10), 360)
            except:
                print(f"WIND SLICE NOT FOUND FOR {wind_time}")
                V = 0
                V_angle = 0
            
            precip_slice = precip_data.sel(time=time, step=step)
            tp = float(precip_slice.tp.values) if 'tp' in precip_slice else 0
            
            df_list.append({
                'timestamp': wind_time,
                'V(m/s)': V,
                'V_angle': V_angle,
                'rainfall(mm)':tp*1000
            })
    df = pd.DataFrame(df_list)
    df = df.set_index('timestamp').sort_index()
    df = df.dropna(axis=0)
    mask = (df.index >= pd.to_datetime('20231109')) & (df.index <= pd.to_datetime('20250904'))
    df = df.loc[mask]
    
    print(df)
    print(df.shape)
    df.to_csv('./results/combined_weather.csv')
    

def plot_era5_csv(file_path='./results/checkpoints/combined_weather.csv', plot_daily=False, plot_storms=False, get_df=False):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)    
    if plot_daily:
        df = df.groupby(pd.to_datetime(df.index).date).agg({'V(m/s)': 'mean', 'rainfall(mm)': 'sum'})
    if get_df: return df
    
    fig, ax = plt.subplots()
    # ax.plot(mdates.date2num(df.index), df['V(m/s)'], label='V(m/s)', color=(0.8, 0.0, 0.0, 0.8))
    ax.bar(mdates.date2num(df.index), df['rainfall(mm)'], label='Rainfall(mm)', width=1.6, color=(0.0, 0.0, 0.8, 0.8))
    # ax2 = ax.twinx()
    # ax2.bar(mdates.date2num(df.index), df['rainfall(mm)'], label='Rainfall(mm)', width=1.6, color=(0.0, 0.0, 0.8, 0.8))
    
    # ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} wind speed (m/s)", color=(0.8, 0.0, 0.0))
    ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} rainfall (mm)")
    # ax2.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} rainfall (mm)", color=(0.0, 0.0, 0.8))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    
    if plot_storms:
        storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
        prev_storm_end = 0
        for storm in storms_data.index:
            dates = storms_data.loc[storm, ['start_date', 'end_date']]
            ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
            # if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
            #     ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
            # else:
            #     ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
            # prev_storm_end = np.datetime64(dates['end_date'])
    
    plt.tight_layout()
    plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_rainfall.eps")


def plot_tidal_data(file_path, t_start:datetime, t_end):
    # df = pd.read_csv(file_path, sep='\s+', skiprows=[0,1,2,3,4,5,6,7,8,10])
    # df.drop('Cycle', axis=1, inplace=True)
    # df.loc[df['ASLVBG02'].str.contains('N'), ['ASLVBG02', 'Residual']] = '-1.0'
    # df[['ASLVBG02', 'Residual']] = df[['ASLVBG02', 'Residual']].apply(lambda x: x.str.strip('M'))
    # df[['ASLVBG02', 'Residual']] = df[['ASLVBG02', 'Residual']].apply(pd.to_numeric)
    # # df['ASLVBG02'] = pd.to_numeric(df['ASLVBG02'].str.strip('M'))
    # df.set_index(pd.to_datetime(df['Date'] + df['Time'].astype(str), format = '%Y/%m/%d%H:%M:%S'), inplace=True)
    
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    if t_end is timedelta:
        t_end = t_start + t_end
    plot_df = df[t_start:t_end]
    
    fig, ax = plt.subplots()
    ax.plot(plot_df.index, plot_df['ASLVBG02'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(12))
    
    ax.set_title('Cromer Tidal Gauge (National Oceanographic Centre)')
    ax.set_ylabel('Tidal Height (m)')
    
    plt.tight_layout()
    # plt.savefig(f'./results/figures/Tidal_Plots/{t_start.date()}_{t_end.date()}_tidal.eps')
    plt.show()


def plot_waverider_csv(file_path:str, plot_daily=False, get_df=False):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    mask = (df.index >= pd.to_datetime('20231109')) & (df.index <= pd.to_datetime('20250904'))
    df = df.loc[mask]
    
    # replacing flagged error values with NaN for later removal/handling (5 unused)
    df.loc[df.Flag == 1, ['Hs(Hm0)(m)','Hmax(m)','Tp(s)','Tz(Tm)(s)','Dirp(degrees)','Spread(deg)']] = np.nan
    df.loc[df.Flag == 2, ['Tp(s)','Dirp(degrees)','Spread(deg)']] = np.nan
    df.loc[df.Flag == 3, ['Dirp(degrees)','Spread(deg)']] = np.nan
    df.loc[df.Flag == 4, ['Spread(deg)']] = np.nan
    df.loc[df.Flag == 6, ['Dirp(degrees)','Spread(deg)']] = np.nan
    df.loc[df.Flag == 7, ['Hs(Hm0)(m)','Hmax(m)','Tp(s)','Tz(Tm)(s)','Dirp(degrees)','Spread(deg)','SST(degC)']] = np.nan
    df.loc[df.Flag == 8, ['SST(degC)']] = np.nan
    df.loc[df.Flag == 9, ['Hs(Hm0)(m)','Hmax(m)','Tp(s)','Tz(Tm)(s)','Dirp(degrees)','Spread(deg)','SST(degC)']] = np.nan
    df = df.dropna(axis=0)
    
    if plot_daily:
        df = df.groupby(pd.to_datetime(df.index).date).agg({'Hs(Hm0)(m)': 'mean', 'Hmax(m)': 'mean', 'Tp(s)': 'mean', 'Tz(Tm)(s)': 'mean', 'Dirp(degrees)': 'mean', 'Spread(deg)': 'mean', 'SST(degC)': 'mean'})
    if get_df: return df
    
    ### Wave height
    # fig = plt.figure(); ax = fig.add_subplot(111)
    # ax.plot(mdates.date2num(df.index), df['Hs(Hm0)(m)'], label='Hs(m)', color=(0.8, 0.0, 0.0))
    # ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} wave height (m/s)", color=(0.8, 0.0, 0.0))
    # fig.autofmt_xdate()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    # plt.tight_layout()
    # plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_wave_height.eps")
    
    ### Wave period
    # fig = plt.figure(); ax = fig.add_subplot(111)
    # ax.plot(mdates.date2num(df.index), df['Tp(s)'], label='Peak wave period (s)', color=(0.8, 0.0, 0.0, 0.4))
    # ax2 = ax.twinx()
    # ax2.plot(mdates.date2num(df.index), df['Tz(Tm)(s)'], label='Zero-crossing wave period (s)', color=(0.0, 0.0, 0.8, 0.4))
    # ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} peak wave period (s)", color=(0.8, 0.0, 0.0))
    # ax2.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} zero-crossing wave period (s)", color=(0.0, 0.0, 0.8))
    # fig.autofmt_xdate()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    # plt.tight_layout()
    # plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_wave_period.eps")
    
    fig = plt.figure(); ax = fig.add_subplot(111)
    wave_f = 2/df['Tp(s)']
    ax.plot(mdates.date2num(df.index), wave_f)
    ax.set_yscale('log')
    ax.set_ylim(0.09, 1.1)
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    plt.tight_layout()
    plt.show()


def plot_met_csv(file_path:str, plot_daily=False, get_df=False):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    mask = (df.index >= pd.to_datetime('20231109')) & (df.index <= pd.to_datetime('20250904'))
    df = df.loc[mask]
    
    # replacing flagged error values with NaN for later removal/handling
    df.loc[df.Flag == 1, ['Wind(m/s)']] = np.nan
    df.loc[df.Flag == 2, ['WindDir(deg)']] = np.nan
    df.loc[df.Flag == 3, ['Baro(hPa)']] = np.nan
    df.loc[df.Flag == 4, ['Tair(degC)']] = np.nan
    df.loc[df.Flag == 5, ['Gust(m/s)']] = np.nan
    df.loc[df.Flag == 6, ['Rainfall(mm)']] = np.nan
    df.loc[df.Flag == 7, ['Solar(W/m^2)', 'UV(W/m^2)']] = np.nan
    df.loc[df.Flag == 8, ['RH(%)']] = np.nan
    df.loc[df.Flag == 9, ['Wind(m/s)', 'WindDir(deg)', 'Gust(m/s)', 'Baro(hPa)', 'Tair(degC)', 'Rainfall(mm)', 'Solar(W/m^2)', 'UV(W/m^2)', 'RH(%)']] = np.nan
    # below are all unrecoreded parameters
    df = df.drop(columns=['Rainfall(mm)', 'Solar(W/m^2)', 'UV(W/m^2)', 'RH(%)'])
    df = df.dropna(axis=0)
    
    if plot_daily:
        df = df.groupby(pd.to_datetime(df.index).date).agg({'Wind(m/s)': 'mean', 'WindDir(deg)': 'mean', 'Gust(m/s)': 'mean', 'Baro(hPa)': 'mean', 'Tair(degC)': 'mean'})
    if get_df: return df
        
    ### Wind & rainfall
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(mdates.date2num(df.index), df['Wind(m/s)'], label='Wind speed (m/s)', color=(0.8, 0.0, 0.0))
    ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} wind speed (m/s)", color=(0.8, 0.0, 0.0))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    plt.tight_layout()
    plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_windspeed.eps")


def plot_nwalsham_rain_csv(get_df=False):
    df = pd.read_csv('results/checkpoints/North-Walsham-rainfall-daily.csv', index_col=2, parse_dates=True)
    df = df[['value']]
    return df


def plot_combined_weather(plot_daily=False, plot_storms=False, t_start=None, t_end=None): 
    df_wave = plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily=plot_daily, get_df=True)
    df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=plot_daily, get_df=True)
    df_era5 = plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily=plot_daily, get_df=True)
    df = pd.concat([df_wave, df_met, df_era5], axis=1)
    
    if t_end is timedelta:
        t_end = t_start + t_end
    df = df[t_start:t_end]
    df = df[[('00:00' or '30:00') in str(s) for s in df.index]]
    
    vars = ['Wind(m/s)','Hs(Hm0)(m)','Tp(s)','Tz(Tm)(s)','rainfall(mm)']
    labels = ['Wind speed (m/s)','Wave height (m)','Peak wave period (s)','Zero-crossing period (s)','Rainfall (mm)']
    
    fig, axs = plt.subplots(len(vars), 1, figsize=(12, 6), sharex=True)
    for ax, col, label in zip(axs, vars, labels):
        if col=='rainfall(mm)':
            ax.bar(mdates.date2num(df.index), df[col])
        else:
            ax.plot(mdates.date2num(df.index), df[col])
        ax.set_ylabel(label)
        ax.grid(axis='x', which='both')
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    
    if plot_storms:
        storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
        for storm in storms_data.index:
            dates = storms_data.loc[storm, ['start_date', 'end_date']]
            axs[-1].axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
    
    plt.subplots_adjust(hspace=0)
    # plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_combined.eps")
    plt.show()


def plot_waverider_direction(daily=False, arrow_every_days=48, arrow_len_deg=20):
    """
    Plot wave direction ('Dirp(degrees)') time series from the waverider csv.
    - arrow_every: place an oriented arrow every N samples (int)
    - arrow_len_deg: visual arrow length in degrees (affects arrow length on the y axis)
    Returns (fig, ax) unless get_df=True (then returns the cleaned dataframe).
    """
    # df_wave = plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily=plot_daily, get_df=True)
    df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=daily, get_df=True)

    times = df_met.index
    wind_dir = df_met['WindDir(deg)'].astype(float).values
    wind_speed = df_met['Wind(m/s)'].astype(float).values
    df_met['WindDir(deg)'].plot(kind='hist')

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, wind_dir, color='tab:blue', lw=0.9)
    ax.set_ylabel('Wave direction (deg)')
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 45))
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    x = mdates.date2num(times)

    if not daily:
        arrow_every_days *= 24
    if arrow_every_days <= 0:
        arrow_every_days = max(1, len(x)//20)
    idx = np.arange(0, len(x), arrow_every_days)
    theta = np.deg2rad(90.0 - wind_dir[idx])
    u = arrow_len_deg * np.cos(theta)
    v = arrow_len_deg * np.sin(theta)

    span_days = (x[-1] - x[0]) if len(x) > 1 else 1.0
    x_scale_days = span_days * 0.01
    u_days = u * (x_scale_days / max(arrow_len_deg, 1.0))
    ax.quiver(x[idx], wind_dir[idx], u_days, v, angles='xy', scale_units='xy', scale=1.0,
              color='k', zorder=5)  # width=0.006, headwidth=4, headlength=6

    compass8 = ['N','NE','E','SE','S','SW','W','NW']
    boundaries = np.arange(-22.5, 360+22.5, 45.0)
    def deg_to_compass(d):
        d = d % 360
        idxc = int(((d + 22.5) % 360) // 45)
        return compass8[idxc]
    for i in idx:
        lbl = deg_to_compass(wind_dir[i])
        card_angle = {'N':0,'NE':45,'E':90,'SE':135,'S':180,'SW':225,'W':270,'NW':315}[lbl]
        if abs(((wind_dir[i] - card_angle + 180) % 360) - 180) <= 15:
            ax.text(times[i], wind_dir[i] + 8, lbl, ha='center', va='bottom', fontsize=8, color='darkred', zorder=6)

    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%Y'))
    plt.tight_layout()
    
    new_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    rose_ax = WindroseAxes.from_ax(theta_labels=new_labels)
    rose_ax.bar(wind_dir, wind_speed, normed=True, bins=np.arange(0, 18, 2))
    rose_ax.set_legend(title = 'Wind Speed in m/s', loc='center right', bbox_to_anchor=(1.25,0.5))
    
    plt.show()


def get_good_wind_periods(daily=False):
    df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=daily, get_df=True)
    
    nw_wind = df_met[df_met['WindDir(deg)'].between(285, 330)]    
    se_wind = df_met[df_met['WindDir(deg)'].between(105, 150)]   
    good_wind = pd.concat([nw_wind, se_wind]).sort_index()
    print(f'{good_wind.shape[0]} of {df_met.shape[0]} || {good_wind.shape[0]/df_met.shape[0]*100}%')
    
    # bad_times = df_met.index.difference(good_wind.index)
    # ax = good_wind['Wind(m/s)'].plot()
    # for t in bad_times:
    #     ax.axvline(t, color='red', alpha=0.3, linewidth=0.8, zorder=2)
    
    deltas = []
    index = good_wind.index.tolist()
    idx = pd.DatetimeIndex(index).sort_values()
    nominal_gap = pd.Timedelta(minutes=10)    # expected regular spacing
    tolerance = pd.Timedelta(minutes=30)      # gaps <= tolerance are treated as continuous

    diffs = idx.to_series().diff().fillna(pd.Timedelta(seconds=0))
    breaks = diffs > tolerance
    group_id = breaks.cumsum()

    windows = []
    singletons = []
    for gid, group in idx.to_series().groupby(group_id):
        times = group.index
        if len(times) == 1:
            singletons.append(times[0])
            continue

        diffs_group = times.to_series().diff().iloc[1:]
        tolerated_mask = (diffs_group > nominal_gap) & (diffs_group <= tolerance)
        n_tolerated = int(tolerated_mask.sum())
        tolerated_total = ( (diffs_group - nominal_gap).where(tolerated_mask, pd.Timedelta(0)) ).sum()

        start = times[0]
        end = times[-1]
        duration = end - start
        windows.append([start, end, duration, n_tolerated, tolerated_total])
    
    windows_df = pd.DataFrame(windows, columns=['start', 'end', 'duration', 'n_tolerated', 'tolerated_total'])
    windows_df.to_csv('./results/checkpoints/wind_windows.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    if 'Wind(m/s)' in df_met.columns:
        ax.plot(df_met.index, df_met['Wind(m/s)'], color='0.8', lw=0.9, label='Wind (m/s)')
    for w in windows:
        s, e = w[0], w[1]
        ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), color='tab:green', alpha=0.25, zorder=2)
    if singletons:
        ylim = ax.get_ylim()
        marker_y = ylim[1] - 0.03 * (ylim[1] - ylim[0])
        ax.plot(singletons, [marker_y]*len(singletons), '|', color='red', markersize=8, label='Singletons')
    ax.set_ylabel('Wind (m/s)')
    ax.set_title('Good-wind windows (highlighted)')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.legend(loc='upper right')
    plt.tight_layout()
    # plt.savefig('./results/checkpoints/wind_windows_timeline.eps', bbox_inches='tight', dpi=120)
    plt.show()
    

def rolling_good_wind_coverage(window_days: int = 3,
                               step_days: int = 1,
                               freq_minutes: int = 10,
                               plot: bool = False):
    """
    Slide a window of `window_days` forward by `step_days` and compute:
      - good_count: number of expected freq_minutes timestamps in the window that meet good-wind
      - observed_slots: number of df_met timestamps inside the window
      - observed_pct: good_count / observed_slots * 100 (NaN if observed_slots==0)

    Returns a pandas.DataFrame indexed by window start (datetime) with the columns above.
    """
    # load full-resolution met (10-min) data
    df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=False, get_df=True)
    if df_met is None or df_met.empty:
        raise RuntimeError("No met data available")

    start = df_met.index.min()
    end = df_met.index.max()
    window_delta = pd.Timedelta(days=window_days)
    step_delta = pd.Timedelta(days=step_days)

    results = []
    current = start
    while current + window_delta <= end + pd.Timedelta(seconds=1):
        win_start = current
        win_end = current + window_delta

        df_window = df_met[win_start:win_end]
        observed_slots = df_window.shape[0]
        observed_good_count = df_window[
            (df_window['WindDir(deg)'].between(285, 330)) # | (df_window['WindDir(deg)'].between(105, 150))
        ].shape[0]

        observed_pct = 100.0 * observed_good_count / observed_slots if observed_slots > 0 else np.nan
        results.append({
            'window_start': win_start,
            'window_end': win_end,
            'observed_good_count': int(observed_good_count),
            'observed_slots': int(observed_slots),
            'observed_pct': float(observed_pct)
        })
        current += step_delta

    df_res = pd.DataFrame(results).set_index('window_start')
    if plot:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df_res.index, df_res['observed_pct'], label='Good % (of observed slots)', marker='x')
        ax.set_ylabel('Percent good wind (%)')
        ax.set_xlabel('Window start')
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    # print(df_res[df_res['observed_pct'] == 0.0])
    df_res = df_res[df_res['observed_pct'] >= 33.0]
    print(df_res)
    # df_res.to_csv('./results/checkpoints/wind_percentages.csv')


def get_rainfall_windows():
    df_met = plot_era5_csv('./results/checkpoints/combined_weather.csv', get_df=True)
    print(df_met.columns)
    df_rain = df_met['rainfall(mm)']
    
    df_rain = df_rain[df_rain >= 2.0]
    print(df_rain.shape)
    df_rain.hist()
    plt.show()
    for i in df_rain.index:
        print(i)


if __name__ == "__main__":
    daily = True
    m = 5
    d0 = datetime(year=2025, month=m, day=1)
    d1 = datetime(year=2025, month=(m+1)%12, day=1)
    # d0 = datetime(year=2024, month=12, day=8); d1 = datetime(year=2024, month=12, day=14)
    # plot_weather()
    # plot_rain_storms()
    # era5_data_to_csv('era5_final.grib')
    # plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily=daily, plot_storms=True)
    # plot_tidal_data('./results/checkpoints/CRO_final.csv', d0, d1)
    
    # plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=daily)
    task_t0 = datetime(year = 2025, month = 2, day = 1, 
                   hour = 0, minute = 0, second = 0, microsecond = 0)
    task_t1 = task_t0 + timedelta(days=14)
    # plot_combined_weather(plot_daily=daily, plot_storms=False, t_start=task_t0, t_end=task_t1)
    plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily=daily)
    
    # plot_waverider_direction(daily=daily, arrow_every_days=7)
    # get_good_wind_periods(daily)
    # rolling_good_wind_coverage(plot=False)
    # get_rainfall_windows()
