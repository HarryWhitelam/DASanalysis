from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
from ppsd_utils import *


### loaders
def load_bgs_data(station, starttime, endtime, component="E"):
    client = Client("http://eida.bgs.ac.uk")
    stream = client.get_waveforms("GB", station, "00", "HH?", starttime, endtime)
    inventory = read_inventory(f"{station.lower()}_station.xml")
    
    return stream.select(component=component), inventory


### bgs example
stream, inventory = load_bgs_data(
    station="BEDF",
    component="E",
    starttime=UTCDateTime("2024-12-11"),
    endtime=UTCDateTime("2024-12-14"),
)

process_ppsd(
    stream,
    inventory,
    title="BEDF",
    outdir="./figures/BEDF",
)