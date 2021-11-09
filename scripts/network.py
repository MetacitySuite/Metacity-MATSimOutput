

import pandas as pd
import xml.etree.ElementTree as ET
import gzip


class network:
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
        self.net.drop(["x","y","modes"], axis=1, inplace=True)


    def return_network(self):
        return self.net


