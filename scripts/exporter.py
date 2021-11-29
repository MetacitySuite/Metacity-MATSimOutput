import os, gc, shutil, json
from multiprocessing import Pool, Manager
from memory_profiler import profile

import geopandas as gpd
import numpy as np
import pandas as pd

import network as net
from agent import MHD, Car, Human

OUTPUT_FORMAT = 'json'
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
        if(self.agent_type == "agent"):
            if(self.export_mode == CARS):
                self.transport_map = self.load_transport_map(['car'])
            else:
                self.transport_map = self.load_transport_map()
        else:
            self.transport_map = {}
        
    

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
    


    def load_transport_chunks(slef, veh_ids, tp_map):
        #lookup map
        chunks_to_load = set()
        for v_id in veh_ids:
            if v_id is not None:
                if v_id.isnumeric():
                    chunks_to_load.add(tp_map["car"][int(v_id)])
                else:
                    veh_type = v_id.split('_')[-1]
                    chunks_to_load.add(tp_map[veh_type][v_id])

        df = pd.DataFrame()
        for file in chunks_to_load:
            ch = pd.read_json(file, lines=True, orient='records')
            df = pd.concat([df,ch])
            df.sort_values("id", kind="stable", inplace=True)
            df.set_index("id", inplace=True)
        #print("Transport loading finished.")
        return df

                

    def load_transport_map(self, vehicle_types = ["car","tram","subway","bus","funicular"]):
        transport_map = {}
        files = os.listdir(OUTPUT+"events/")

        for veh in vehicle_types:            
            if veh+"_map.json" in files:
                print(OUTPUT+"events/"+veh+"_map.json")
                vehicle_map = pd.read_json(OUTPUT+"events/"+veh+"_map.json", typ="series")
                vehicle_map = vehicle_map.apply(lambda x: OUTPUT+"events/"+veh+"/"+str(x)+".json") #updates to full path to chunk
                transport_map[veh] = vehicle_map
                
        print("Transport map (car):",transport_map["car"])
        return transport_map


    def link_network(self, df):
        #join links and coordinates
        df = df.merge(self.network.set_index("link"), on='link', how="left").fillna(value=np.nan)
        if("coords_to" in df.columns):
            df = df.drop(columns=["coords_from", "coords_to"])

        df["coords_to"] = self.return_coords(df.coords_x, df.coords_y, df.x_to, df.y_to)
        df['coords_from'] = self.return_coords(df.coords_x,df.coords_y, df.x_from, df.y_from) 
        return df


    def pick_vehicle_events(self, df, v_id, transport):
        veh_events = pd.DataFrame()

        veh_row = transport.loc[int(v_id)]
        vehicle = pd.DataFrame.from_dict(veh_row["events"])

        starts = df.iloc[np.where(df.vehicle_id == v_id)].loc[np.where(df.type == "PersonEntersVehicle")].time.to_list()
        ends = df.iloc[np.where(df.vehicle_id == v_id)].loc[np.where(df.type == "PersonLeavesVehicle")].time.to_list()

        vehicle["vehicle_id"] = v_id
        for start,dest in zip(starts,ends):
            veh_events = veh_events.append(vehicle.iloc[np.where((vehicle["time"] >= start) & (vehicle["time"]<= dest))])

        return veh_events


    def append_vehicles(self, df, agent_id, vehicle_ids, tp_map, verbal=False):
        events = pd.DataFrame()
        #load chunks for vehicle_ids to RAM
        transport = self.load_transport_chunks(vehicle_ids, tp_map)
        for v_id in vehicle_ids:
            if v_id is not None:
                if(verbal):
                    print("Vehicle id:",v_id)

                veh_events = self.pick_vehicle_events(df, v_id, transport)

                if('type' in veh_events.columns):
                    drop_idx = veh_events[
                                (veh_events['type'] == "PersonEntersVehicle") | #& (veh_events['person_id'] != str(agent_id))) | #
                                (veh_events['type'] == "PersonLeavesVehicle") | #& (veh_events['person_id'] != str(agent_id))) | #
                                (veh_events['type'] == "VehicleDepartsAtFacility") |
                                (veh_events['type'] == "VehicleArrivesAtFacility") |
                                (veh_events['type'] == "vehicle leaves traffic") | 
                                (veh_events['type'] == "vehicle enters traffic") |
                                (veh_events['type'] == "left link")
                                ].index      
                    veh_events.drop(drop_idx, inplace=True)

                events = events.append(veh_events)
                del veh_events

        del transport #!!!
        
        df = df.append(events, ignore_index=True)
        del events
        gc.collect()
        df = df.sort_values(["time"], kind="stable") #, "type"
        df.reset_index(inplace=True)
        if("index" in df.columns):
            df.drop(columns=["index"], inplace=True)
        return df

    def other_transport(self, vehicle_ids):
        #check if vehicle_ids contains other transport than cars
        for v in vehicle_ids:
            if v != None and not v.isnumeric():
                return True
        
        return False

    
    def link_transport(self,df, agent_id, tp_map):
        vehicle_ids = [ x for x in list(df.vehicle_id.unique())] #veh ids
        #print("Vehicle ids:", vehicle_ids)
        if len(vehicle_ids) == 1:
            return pd.DataFrame()
        
        if self.export_mode == CARS and self.other_transport(vehicle_ids):
            return pd.DataFrame() #omitting this agent 
            
        df = self.append_vehicles(df, agent_id, vehicle_ids, tp_map, verbal=False)
        df["person_id"] = agent_id
        return df


    
    def prepare_agent(self, row, agent_id, tp_map, verbal=False):
        #prep agent
        v = pd.DataFrame.from_dict(row["events"])

        #join links and coordinates
        if self.agent_type == "agent":
            v = self.link_transport(v, agent_id, tp_map) 

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
        agent.extract_trips(verbal)
        return agent


    def extract_chunk(self, chunk, tp_map, output_type, path, verbal=False):
        path = path + ".sim"
        
        if(output_type == 'shp'):
            output = gpd.GeoDataFrame()
        else:
            output = []
        
        for i,row in chunk.iterrows(): #for each agent in chunk
            a = self.prepare_agent(row,row.id, tp_map, verbal)
            if not (a is None):
                a.prepare_geotrips(output_type)
                if(output_type == 'shp'):
                    output = output.append(a.geotrips.copy())
                else:
                    output.extend(a.geotrips)
        del a

        #save chunk
        if(output_type == 'shp' and not output.empty):
            #reset trip_index
            output.reset_index(inplace=True)
            #save GeoDataFrame as .SHP
            output.to_file(filename=path)

        elif(output_type == "json"):
            #save list of agents to JSON
            with open(path, 'w') as f:
                json.dump(output, f)
                f.close()

        del output
        del chunk
        gc.collect()


    def chunk_task(self, args):
        chunk_path, path_prefix, form, tp_map = args
        print("Loading:",chunk_path)
        chunk = pd.read_json(chunk_path, lines=True, orient='records')
        chunk_i = int(chunk_path.split('/')[-1].split('.')[0])
        path =  path_prefix+self.agent_type+'_sec_'+str(chunk_i)
        self.extract_chunk(chunk, tp_map, output_type = form, path=path, verbal=False)
        print("Chunk saved to:",path)


    def parallel_run(self, files, proc, path_prefix, format):

        args = list()
        for f in files:
            args.append([f, path_prefix, format, self.transport_map])

        with Pool(proc) as pool:
            pool.map(self.chunk_task, args)

        pool.close()
        pool.join()
        

    def export_agents(self, format = OUTPUT_FORMAT, parallel=PARALLEL, proc=os.cpu_count()):
        path_prefix = OUTPUT+'matsim_agents_'+str(format)+'/'+self.agent_type+'/'   
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        else:
            shutil.rmtree(path_prefix)
            os.makedirs(path_prefix)
            
        print("Saving to:", path_prefix, os.path.exists(path_prefix))

        dirc = OUTPUT+"events/"+self.agent_type
        files = [ dirc+"/"+f for f in os.listdir(dirc)]
        chunk_i = 0
        if(parallel):
            self.parallel_run(files, proc, path_prefix, format)

        else:
            print("# files:", len(files))
            for file in files:
                self.chunk_task([file,path_prefix,format, self.transport_map])
                chunk_i +=1
        return
        



    

