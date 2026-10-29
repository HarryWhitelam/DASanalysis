from obspy import read
from datetime import datetime
import pickle
import numpy as np

class SegyReader:
    file_path: str
    data: np.ndarray
    properties: dict
    fileinfo: dict
    
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.data = self.read_das_file()
        self.properties = self.get_dir_properties()
        self.fileinfo = self.get_dir_fileinfo()
    
    def get_data(self, cha1, cha2):
        return self.data[:, cha1:cha2+1]
    
    def get_properties(self):
        return self.properties
    
    def get_file_info(self):
        return self.fileinfo
    
    
    def get_dir_properties(self):
        path, fn = self.file_path.rsplit('/', 1)
        path += '/properties.p'
        
        with open(path, 'rb') as props_path:
            props:dict = pickle.load(props_path)
        props.update({'GPSTimeStamp': datetime.strptime(fn.split('UTC')[-1].split('.')[0].replace('_', ''), '%Y%m%d%H%M%S')})
        
        return props
    
    
    def get_dir_fileinfo(self):
        path, _ = self.file_path.rsplit('/', 1)
        path += '/fileinfo.p'
        
        with open(path, 'rb') as fileinfo_path:
            fileinfo:dict = pickle.load(fileinfo_path)
        
        return fileinfo


    def read_das_file(self):        
        stream = read(self.file_path)
        stats = stream[0].stats
        
        npts = stats.npts           # temporal axis
        ncha = len(stream)          # spatial axis
        data = np.empty(shape=(npts, ncha))
                
        for count, trace in enumerate(stream):
            data[:, count] = trace.data
        
        return data

    def _read_properties(self):
        ### This is a redundant function, the TdmsReader needs to run this before props can be added. 
        ### This function PURELY exists to prevent errors. It offers NO functionality at present. 
        return
