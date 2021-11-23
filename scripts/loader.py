import pandas as pd
import xml.etree.ElementTree as ET
import gzip
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os

from memory_profiler import profile
from multiprocessing import Pool

#pd.set_option('display.max_columns', None)
#warnings.filterwarnings('ignore')

XML = "../../pop10k-eh16-qsim4-100it/output_events.xml.gz"
CSV = "./../output/population/"


events_dtypes = {
    "Unnamed: 0" : np.float64,
    "time": np.float64,
    "type": str,
    "driverId": str,
    "vehicleId": str,
    "transitLineId": str,
    "transitRouteId": str,
    "departureId": str,
    "person": str,
    "link": str,
    "legMode": 'category', #category
    "vehicle": str,
    "networkMode": str, #category
    "relativePosition": np.float64,
    "facility": str,
    "delay": np.float64,
    "x": np.float64,
    "y": np.float64,
    "actType": str,
    "computationalRoutingMode": str,
    "distance" : np.float64,
    "mode": str,
    "agent": str,
    "atStop": str
}


def clear_directory(directory : str):
    """
    Creates directory if it does not exists and clears an existing directory.

    Args:
        directory (str): Path to directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)


class EventLoader:

    def __init__(self, xml_path = XML, csv_path =  CSV):
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        self.xml_path = xml_path
        self.csv_path = csv_path 

    def load_population_CSV(self, chunk_size=10e6, el_tag="event"):
        """
        Loads MATSim output_events.xml to separate CSV files for further porcessing.

        Args:
            chunk_size (int): Number of events in one CSV file.
            el_tag (str): Name of XML tag to be extracted.
        """
        clear_directory(self.csv_path)
    
        chunk_i=0
        dict_list = []
        elem_count = 0

        if(self.xml_path.split('.')[-1] == "gz"):
            source = gzip.open(self.xml_path)
        else: source = self.xml_path

        for _, elem in ET.iterparse(source, events=("end",)):
            if elem.tag == el_tag:
                dict_list.append(elem.attrib)      #PARSE ALL ATTRIBUTES
                elem.clear()
                elem_count +=1

            if(elem_count % chunk_size == 0):
                df = pd.DataFrame(dict_list)
                dict_list = []
                df.to_csv(self.csv_path+str(chunk_i)+".csv")
                del df
                chunk_i +=1

        #print("Saving chunk:", chunk_i)
        df = pd.DataFrame(dict_list)
        df.to_csv(self.csv_path+str(chunk_i)+".csv")
        print("CSV files saved to:", self.csv_path, chunk_i+1,"files")
        return

    def read_csv(self, path):
        return pd.read_csv(path, dtype=events_dtypes)

    def gather_CSV_files(self):
        csv_files = list()

        for csv in os.listdir(self.csv_path):
            csv_files.append(self.csv_path+csv)

        print("Files prepared:", len(csv_files), "files")
        return csv_files








