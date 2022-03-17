

import pandas as pd
import xml.etree.ElementTree as ET
import gzip
import geopandas as gpd
from shapely.geometry import LineString


class Network:
    def __init__(self):
        self.xml_path = ""
        self.links = pd.DataFrame()
        self.nodes = pd.DataFrame()
        self.net = pd.DataFrame()

    def set_path(self, p):
        self.xml_path = p

    def status(self):
        print("Network status:")
        print("path:", self.xml_path)
        print("links:",self.links.shape[0])
        print("nodes:",self.nodes.shape[0])
        print("network:",self.net.shape[0])

    def load_elements(self, el_tag):
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

        df = pd.DataFrame(dict_list)
        df[el_tag] = df["id"]
        df.drop(["id"], axis=1, inplace=True)
        return df


    def load_links(self):
        self.links = self.load_elements(el_tag='link')
        

    def load_nodes(self):
        self.nodes = self.load_elements(el_tag='node')
        self.nodes.set_index("node")


    def join_network(self):
        self.net = self.links.join(self.nodes.set_index("node"), on="from")
        self.net = self.net.join(self.nodes.set_index("node"), on="from", rsuffix="_from")
        self.net = self.net.join(self.nodes.set_index("node"), on="to", rsuffix="_to")
        self.net["link_modes"] = self.net.modes
        self.net["freespeed"] = self.net.freespeed
        self.net["capacity"] = self.net.capacity
        self.net["lanes"] = self.net.permlanes
        self.net.drop(["x","y","modes"], axis=1, inplace=True)


    def return_network(self):
        return self.net

    def export_shp(self, path_shp):
        #todo
        lines = []
        modes = []
        speeds = []
        capacities = []
        lanes = []
        idx = []
        for i,link in self.net.iterrows():
            line = LineString([(-float(link['x_from']), -float(link['y_from'])), (-float(link['x_to']), -float(link['y_to']))])
            lines.append(line)
            idx.append(link["link"])
            modes.append(link['link_modes'])
            speeds.append(link['freespeed'])
            capacities.append(link['capacity'])
            lanes.append(link['lanes'])


        network_shp = gpd.GeoDataFrame(data={
                                'geometry': lines,
                                'mode' : modes,
                                'speed' : speeds,
                                'capacity' : capacities,
                                'index' : idx,
                                })

        #save GeoDataFrame as .SHP
        network_shp.to_file(filename=path_shp)
        del network_shp

        



