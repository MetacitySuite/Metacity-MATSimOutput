import json
from multiprocessing import Pool, Process
import os, sys, gc

import geopandas as gpd
import numpy as np
import pandas as pd

import network as net
from agent import MHD, Car, Human


CHUNK_SIZE = 500
OUTPUT_FORMAT = 'shp'
PARALLEL = True
OUTPUT = "./../output/"

class Exporter:
    def __init__(self, agent_type, network_path):
        self.load_network(network_path)
        self.set_agent(agent_type)
        if(agent_type == "agent"):
            self.load_transport()

    def set_agent(self, type):
        self.agent_type = type

    def count_agents(self):
        self.events = pd.read_json(OUTPUT+"events/"+self.agent_type+".json") #memory hog
        print("Loaded agents", self.agent_type,"#:", self.events.shape[0])
        self.agent_count = self.events.shape[0]
        del self.events
        gc.collect()
    

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

    def load_transport(self, vehicle_types = ["car","tram","subway","bus","funicular"]):
        #load all transport event files for later linking (now about 9GB in RAM :3 sorry future me)
        self.transport = {}

        for file in os.listdir(OUTPUT+"events/"):
            if file.split('.')[0] in vehicle_types:
                print("Loading:", file)
                df = pd.read_json(OUTPUT+"events/"+file, lines=True, orient='records')
                df.sort_values("id", kind="stable", inplace=True)
                df.set_index("id", inplace=True)
                self.transport[str(file.split('.')[0])] = df.copy()
                del df
        print("Transport loaded:", sys.getsizeof(self.transport)/1000)

    def link_network(self, df):
        #join links and coordinates
        df = df.join(self.network.set_index("link"), on='link').fillna(value=np.nan)
        if("coords_to" in v.columns):
            df = df.drop(columns=["coords_from", "coords_to"])

        df["coords_to"] = self.return_coords(df.coords_x, df.coords_y, df.x_to, df.y_to)
        df['coords_from'] = self.return_coords(df.coords_x,df.coords_y, df.x_from, df.y_from) 
        return df

    def append_vehicles(self, df, vehicle_ids, verbal=False):
        events = pd.DataFrame()
        for v_id in vehicle_ids:
            if v_id is not None:
                veh_events = pd.DataFrame()
                if(verbal):
                    print("Vehicle id:",v_id)

                if(str(v_id).isnumeric()):
                    if(verbal):
                        print("\tVehicle type:", "car")
                    car_row = self.transport['car'].loc[int(v_id)]
                    car = pd.DataFrame.from_dict(car_row["events"])

                    starts = df.iloc[np.where(df.vehicle_id == v_id)].loc[np.where(df.type == "PersonEntersVehicle")].time.to_list()
                    ends = df.iloc[np.where(df.vehicle_id == v_id)].loc[np.where(df.type == "PersonLeavesVehicle")].time.to_list()

                    car["vehicle_id"] = v_id
                    for start,dest in zip(starts,ends):
                        if(verbal):
                            print(start, dest)

                        veh_events = veh_events.append(car.iloc[np.where((car["time"] >= start) & (car["time"]<= dest))])
                elif  v_id.split('_')[-1] in self.transport.keys():
                    veh_type = v_id.split('_')[-1]
                    if(verbal):
                        print("\tVehicle type:", veh_type)
                    veh_row = self.transport[veh_type].loc[v_id]
                    vehicle = pd.DataFrame.from_dict(veh_row["events"])
                    vehicle["vehicle_id"] = v_id

                    #tohle je spatne # tak uz dobry
                    a = df.iloc[np.where(df.vehicle_id == v_id)] #pick start and end points of vehicle interaction(s)
                    starts = a.iloc[np.where(a.type == "PersonEntersVehicle")].time.to_list()
                    ends = a.iloc[np.where(a.type == "PersonLeavesVehicle")].time.to_list()
                    for start,dest in zip(starts,ends):
                        if(verbal):
                            print(start, dest)
                            display(vehicle.iloc[np.where((vehicle["time"] >= start) & (vehicle["time"]<= dest))])
                        veh_events = veh_events.append(vehicle.iloc[np.where((vehicle["time"] >= start) & (vehicle["time"]<= dest))])

                #drop events
                if('type' in veh_events.columns):
                    drop_idx = veh_events[
                                ((veh_events['type'] == "PersonEntersVehicle") & (veh_events['person_id'] != str(self.id))) | #
                                ((veh_events['type'] == "PersonLeavesVehicle") & (veh_events['person_id'] != str(self.id))) | #
                                #(veh_events['type'] == "departure") | 
                                #(veh_events['type'] == "arrival") | 
                                (veh_events['type'] == "vehicle leaves traffic") | 
                                (veh_events['type'] == "vehicle enters traffic") |
                                (veh_events['type'] == "left link")
                                ].index      
                    veh_events.drop(drop_idx, inplace=True)

                events = events.append(veh_events)
        df = df.append(events, ignore_index=True)
        df = self.link_network(df)
        df = df.sort_values(["time"], kind="stable") #, "type"
        return df


    def link_transport(self,df):
        if(self.agent_type != "agent"):
            return
        vehicle_ids = [ x for x in list(df.vehicle_id.unique())] #veh ids
        #print("Vehicle ids:", vehicle_ids)
        df = self.append_vehicles(df, vehicle_ids, verbal=False)
        df.reset_index()
        df.drop(columns=["index"], inplace=True)
        drop_idx = df[ 
            (df['type'] == "vehicle leaves traffic") | 
            (df['type'] == "vehicle enters traffic") |
            (df['type'] == "left link")
            ].index
        df.drop(drop_idx, inplace=True)
        df["person_id"] = self.id
        return df

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
        v = self.link_transport(v)
        v = self.link_network(v)

        if(self.agent_type == "car"):
            v.drop(["from","to","length","event_id","permlanes",'link_modes','networkMode','legMode','relativePosition'], axis=1, inplace=True)
            agent = Car(self.agent_type, id)
        elif(self.agent_type == "agent"):
            v.drop(["from","to","length","event_id","permlanes",'link_modes','networkMode','legMode','relativePosition'], axis=1, inplace=True)
            agent = Human(self.agent_type, id)
        else:
            v.drop(["from","to","length","event_id","permlanes",'link_modes','atStop','destinationStop','departure','networkMode','legMode','relativePosition'], axis=1, inplace=True)
            agent = MHD(self.agent_type, id)

        
        agent.set_events(v)
        print("Memory (kB):",v.memory_usage(index=True).sum()/1000)
        del v
        agent.extract_trips(verbal) #todo Human trips
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
            gc.collect()

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
        gc.collect()


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
        



    

