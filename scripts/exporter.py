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
CARS = "cars_only"
ALL = "all_transport"
ALL_WALK = "all_with_walk"

class Exporter:
    def __init__(self, agent_type, network_path, export_mode):
        self.load_network(network_path)
        self.agent_type = agent_type
        self.export_mode = export_mode
        if(agent_type == "agent"):
            if(self.export_mode == CARS):
                self.load_transport(['car'])
            else: #load all
                self.load_transport()


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
        print("Transport loading finished.")

    def link_network(self, df):
        #join links and coordinates
        df = df.merge(self.network.set_index("link"), on='link', how="left").fillna(value=np.nan)
        if("coords_to" in df.columns):
            df = df.drop(columns=["coords_from", "coords_to"])
        #display(df.head(2))

        df["coords_to"] = self.return_coords(df.coords_x, df.coords_y, df.x_to, df.y_to)
        df['coords_from'] = self.return_coords(df.coords_x,df.coords_y, df.x_from, df.y_from) 
        return df

    def append_vehicles(self, df, agent_id, vehicle_ids, verbal=False):
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
                        veh_events = veh_events.append(car.iloc[np.where((car["time"] >= start) & (car["time"]<= dest))])
                
                elif  v_id.split('_')[-1] in self.transport.keys():
                    veh_type = v_id.split('_')[-1]
                
                    veh_row = self.transport[veh_type].loc[v_id]
                    vehicle = pd.DataFrame.from_dict(veh_row["events"])
                    vehicle["vehicle_id"] = v_id

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
                                ((veh_events['type'] == "PersonEntersVehicle") & (veh_events['person_id'] != str(agent_id))) | #
                                ((veh_events['type'] == "PersonLeavesVehicle") & (veh_events['person_id'] != str(agent_id))) | #
                                (veh_events['type'] == "VehicleDepartsAtFacility") |
                                (veh_events['type'] == "VehicleArrivesAtFacility")
                                #(veh_events['type'] == "vehicle leaves traffic") | 
                                #(veh_events['type'] == "vehicle enters traffic") |
                                #(veh_events['type'] == "left link")
                                ].index      
                    veh_events.drop(drop_idx, inplace=True)

                events = events.append(veh_events)
        df = df.append(events, ignore_index=True)
        #df = self.link_network(df)
        df = df.sort_values(["time"], kind="stable") #, "type"
        return df

    def other_transport(self, vehicle_ids):
        #check if vehicle_ids contains other transport than cars
        for v in vehicle_ids:
            if v != None and not v.isnumeric():
                return True
        
        return False

    def link_transport(self,df, agent_id):
        vehicle_ids = [ x for x in list(df.vehicle_id.unique())] #veh ids
        #print("Vehicle ids:", vehicle_ids)

        if len(vehicle_ids) == 1:
            return pd.DataFrame()
        
        if self.export_mode == CARS and self.other_transport(vehicle_ids):
            return pd.DataFrame()
            

        df = self.append_vehicles(df, agent_id, vehicle_ids, verbal=False)
        df.reset_index()
        if("index" in df.columns):
            df.drop(columns=["index"], inplace=True)
        df["person_id"] = agent_id
        return df

    def prepare_agent(self, row, agent_id, verbal=False):
        #prep agent
        v = pd.DataFrame.from_dict(row["events"])

        #join links and coordinates
        if self.agent_type == "agent":
            v = self.link_transport(v, agent_id)

        if(v.empty):
            del v
            return None
            
        v = self.link_network(v)
        #remove unused event types
        drop_idx = v[
            (v['type'] == "vehicle leaves traffic") | 
            (v['type'] == "vehicle enters traffic") |
            (v['type'] == "left link")
            ].index
        v.drop(drop_idx, inplace=True)

        cols = ["from","to","length","event_id",
        "permlanes",'link_modes','atStop','destinationStop',
        'departure','networkMode','legMode','relativePosition']

        to_del = []
        for col in cols:
            if(col in v.columns):
                to_del.append(col)

        v.drop(to_del, axis=1, inplace=True)
            
        if(self.agent_type == "car"):
            agent = Car(self.agent_type, agent_id)
        elif(self.agent_type == "agent"):
            agent = Human(self.agent_type, agent_id)
        else:
            agent = MHD(self.agent_type, agent_id)
        
        agent.set_events(v)
        #print("Memory (kB):",v.memory_usage(index=True).sum()/1000)
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
            if a != None and (output_type == 'shp'):
                a.prepare_geotrips()
                output = output.append(a.geotrips.copy())
            #else:
            #    print("implement (geo)json support")
            #    return
            del a

        #save chunks
        if(output_type == 'shp' and not output.empty):
            #reset trip_index
            output.reset_index(inplace=True)
            #save GeoDataFrame as .SHP
            output.to_file(filename=path)

        elif(output_type == "json"):
            print("implement (geo)json support")
            with open(path, 'w') as f:
                json.dump(output, f)
                f.close()

        del output
        del chunk
        gc.collect()


    def chunk_task(self, chunk_i, ids, path_prefix, format):
        #path =  path_prefix+self.agent_type+'_sec_'+str(chunk_i)+'.'+format
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
        



    

