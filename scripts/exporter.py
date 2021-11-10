from os import path
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import network as net
from agent import Agent
from multiprocessing import Pool, Process

CHUNK_SIZE = 500
OUTPUT_FORMAT = 'shp'
PARALLEL = True
OUTPUT = "./../output/"

class Exporter:
    def __init__(self, agent_type, network_path):
        self.load_network(network_path)
        self.set_agent(agent_type)

    def set_agent(self, type):
        self.agent_type = type

    def count_agents(self):
        self.events = pd.read_json(OUTPUT+"events/"+self.agent_type+".json") #memory hog
        print("Loaded agents", self.agent_type,"#:", self.events.shape[0])
        self.agent_count = self.events.shape[0]
        del self.events
    

    def load_network(self, network_path):
        n = net.Network()
        n.set_path(network_path)
        n.load_nodes()
        n.load_links()
        n.join_network()
        n.status()
        self.network = n.return_network()
        print("Memory (kB):",self.network.memory_usage(index=True).sum()/1000)
        

    def return_coords(self, coords_x, coords_y, i_x, i_y):
        new_coords = [ [-float(x), - float(y)]
                                if not np.isnan(x)
                                else [-float(a), -float(b)] 
                                for x,y,a,b in zip(coords_x, coords_y, i_x, i_y)]
        return new_coords


    def load_events(self): #unused: memory hog
        self.events = pd.read_json(OUTPUT+"events/"+self.agent_type+".json")
        print("Loaded agents", self.agent_type,"#:", self.events.shape[0])
        display(self.events.info())
        self.events.sort_values("id", kind="stable", inplace=True)
        self.events.set_index("id", inplace=True)
        print("Memory (kB):",self.events.memory_usage(index=True).sum()/1000)


    def load_events_chunk(self, chunk_size):
        reader = pd.io.json.read_json(OUTPUT+"events/"+self.agent_type+".json", lines=True, orient='records', chunksize=chunk_size)
        return reader


    def prepare_agent(self, row, id, verbal=False):
        #prep agent
        v = pd.DataFrame.from_dict(row["events"])
        #remove unused event types
        drop_idx = v[
            (v['type'] == "vehicle leaves traffic") | 
            (v['type'] == "vehicle enters traffic") |
            (v['type'] == "left link")
            ].index
        v.drop(drop_idx, inplace=True)
        #join links and coordinates
        v = v.join(self.network.set_index("link"), on='link').fillna(value=np.nan)
        if(self.agent_type != "car"):
            v.drop(["from","to","length","event_id","permlanes",'link_modes','atStop','destinationStop','departure','networkMode','legMode','relativePosition'], axis=1, inplace=True)
        else: # 'atStop' 'destinationStop' 'departure'
            v.drop(["from","to","length","event_id","permlanes",'link_modes','networkMode','legMode','relativePosition'], axis=1, inplace=True)

        if("coords_to" in v.columns):
            v = v.drop(columns=["coords_from", "coords_to"])

        v["coords_to"] = self.return_coords(v.coords_x, v.coords_y, v.x_to, v.y_to)
        v['coords_from'] = self.return_coords(v.coords_x,v.coords_y, v.x_from, v.y_from) 

        agent = Agent(self.agent_type, id)
        agent.set_events(v)
        print("Memory (kB):",v.memory_usage(index=True).sum()/1000)
        if(self.agent_type != "car"):
            agent.extract_trips(verbal) #todo
        else:
            agent.extract_trips_cars(verbal) #todo
        return agent


    def extract_chunk(self, chunk, output_type, path, verbal=False):
        if(output_type == 'shp'):
            output = gpd.GeoDataFrame()
        else:
            output = []
        
        for i,row in chunk.iterrows(): #for each agent in chunk
            a = self.prepare_agent(row,row.id,verbal)
            if(output_type == 'shp'):
                a.prepare_geotrips()
                output = output.append(a.geotrips.copy())
                
            else:
                #a.prepare_json()
                #output.append(a.jsontrips)
                print("implement (geo)json support")
                return
            del a

        #save chunks
        if(output_type == 'shp'):
            #reset trip_index
            output.reset_index(inplace=True)
            #save GeoDataFrame as .SHP
            output.to_file(filename=path)

        else:
            with open(path, 'w') as f:
                json.dump(output, f)
                f.close()

        del output
        del chunk



    def chunk_task(self, chunk_i, ids, path_prefix, format):
        path =  path_prefix+self.agent_type+'_sec_'+str(chunk_i)+'.'+format
        self.extract_chunk(ids, output_type = format, path=path, verbal=False)
        print("Chunk saved to:",path)

        
    def export_agents(self, chunk_size = CHUNK_SIZE, format = OUTPUT_FORMAT, parallel=PARALLEL):
        path_prefix = OUTPUT+'matsim_vehicles_'+str(format)+'/chunks'+str(chunk_size)+'/'   
        event_reader = self.load_events_chunk(chunk_size) 
        chunk_i = 0
        if(parallel):
            p_list = []
            for chunk in event_reader:
                # Instantiates the thread
                print("Agents in chunk", chunk_i, ":", chunk.shape[0])
                p = Process(target=self.chunk_task, args=(chunk_i, chunk, path_prefix, format))
                chunk_i+=1
                p.start()
                p_list.append(p)

            for t in p_list:
                t.join()

        else:
            for chunk in event_reader:
                print("Agents in chunk", chunk_i, ":", chunk.shape[0])
                self.chunk_task(chunk_i,chunk,path_prefix,format)
                chunk_i +=1

        return
        



    

