from obspy import Stream, read, read_inventory
from ppsd_utils import *

def load_local_data(station, component, dates, mseed_dir, station_dir):
    stream = Stream()
    for date in dates:
        stream += read(f"{mseed_dir}/5J.{station}..DP{component}..{date}T00:00:00.000000Z.MSEED")
    inventory = read_inventory(f"{station_dir}/5J.{station}..DP{component}.76_6")
    
    return stream, inventory

### geophone example
stations = ["04915", "05100"]
component = "E"

dates = [
    f"2025-04-{day:02d}"
    for day in range(1, 11)
]

for station in stations:

    stream, inventory = load_local_data(
        station=station,
        component=component,
        dates=dates,
        mseed_dir="/data/localraid/UEA_nodes_mseed_mV/rotated",
        station_dir="/data/localraid/UEA_nodes_mseed_mV/stations",
    )

    process_ppsd(
        stream,
        inventory,
        title=f"Geophone {station}",
        outdir=f"./figures/{station}",
    )

