import pandas as pd
import xml.etree.ElementTree as ET
import gzip
import os
import numpy as np
import pandas as pd
import os
import shutil
import json
import warnings
import gc
from memory_profiler import profile
from multiprocessing import Pool

import loader as csv_loader

#pd.set_option('display.max_columns', None)
#warnings.filterwarnings('ignore')

EVENTS = "./../output/events/"
AGENTS = "./../output/agents/"

vehicle_types = ["bus","car","funicular","subway", "tram"]
vehicle_types = ["car"]


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



class Chunk:
    def __init__(self, agent_id = "", events = []):
        self.id = agent_id
        self.events = events

    def append_events(self,events):
        self.events.extend(events)

    def return_dict(self):
        return {
            "id" : self.id,
            "events" : self.events
        }
    
    def save_chunk(self, path, file):
        if not os.path.exists(path):
            os.makedirs(path)
        #open load and append
        if(os.path.isfile(path+file)):
            with open(path+file, 'r') as f:
                saved = json.load(f)

            saved["events"].extend(self.events)
            with open(path+file, 'w') as f:
                json.dump(saved,f)
                           
        else:
            with open(path+file, 'w') as f:
                if(os.path.getsize(path+file) == 0):
                    json.dump(self.return_dict(),f)
        return



def load_agent_events(row):
    event = {}
    event["event_id"] = row[0]
    event["time"] = row["time"]
    event["type"] = row["type"]
    event["link"] = row["link"]
    event["vehicle_id"] = row["vehicle"]
    event["delay"] = row["delay"]
    event["actType"] = row["actType"]
    event["legMode"] = row["legMode"]
    event["coords_x"] = row["x"]
    event["coords_y"] = row["y"]
    return event

def load_vehicle_events(row, vehicle_type):
    event = {}
    event["event_id"] = row[0]
    event["time"] = row["time"]
    event["type"] = row["type"]
    event["link"] = row["link"]
    event["person_id"] = row["person"]
    event["delay"] = row["delay"]
    event["facility"] = row['facility']

    if isinstance(row['facility'],str):
        event['link'] = row['facility'].split(":")[-1]
    
    event["networkMode"] = row['networkMode']
    event["relativePosition"] = row['relativePosition']
    event["actType"] = row["actType"]
    event["legMode"] = row["legMode"]
    event["coords_x"] = row["x"]
    event["coords_y"] = row["y"]

    if(vehicle_type != "car"):
        if(event["type"] == "TransitDriverStarts"):
            event["transitLine"] = row['transitLineId']
            event["transitRoute"] = row['transitRouteId'] ## add to output
        event["departure"] = row['departureId']
        event["atStop"] = row["atStop"]
        event["destinationStop"] = row["destinationStop"]
    return event


class EventParser:
    def __init__(self, csv_files, agents_path = AGENTS):
        self.csv_files = csv_files
        self.agents_path = agents_path # 

    def __call__(self):
        self.clear_directory(self.agents_path+"/agent")
        for veh_type in vehicle_types:
            self.clear_directory(self.agents_path+"/"+veh_type)
        
        for csv in self.csv_files:
            self.load_agents_from_population(csv)


    def process_agent(self, person):
        agent_id = person.person.unique()[0]
        agent = person.sort_values("time")
        
        events = []
        chunk = Chunk(agent_id)
        for i, row in agent.iterrows():
            events.append(load_agent_events(row))

        chunk.append_events(events)
        chunk.save_chunk(self.agents_path+"/agent","/"+str(agent_id)+".json")
        return


    def process_vehicle(self, args):
        vehicle_df, vehicle_type = args
        vehicle = vehicle_df.sort_values("time")
        events = []
        
        if(vehicle_type == "car"):
            vehicle_id = int(vehicle.vehicle.unique()[0])
        else:
            vehicle_id = vehicle.vehicle.unique()[0]
    
        chunk = Chunk(vehicle_id)
    
        for i, row in vehicle.iterrows():
            events.append(load_vehicle_events(row, vehicle_type))
    
        chunk.append_events(events)
        print("Saving:", vehicle_id)
        chunk.save_chunk(self.agents_path+"/"+vehicle_type,"/"+str(vehicle_id)+".json")
        return
    
    def save_vehicles_parallel(self, args, cpus):
        print("Number of vehicles in loaded chunk:", len(args))
        if(len(args) < 1):
            return
        print("processing:")
        if(cpus == 1):
            print("sequential")
            for arg in args:
                self.process_vehicle(arg)
            return

        with Pool(cpus) as pool:
            pool.map(self.process_vehicle, args)

        pool.close()
        pool.join()
        return

    def save_agents_parallel(self, persons, cpus):
        print("Number of agents in loaded chunk:", len(persons))
        if(len(persons) < 1):
            return
        print("processing:")
        with Pool(cpus) as pool:
            pool.map(self.process_agent, persons)

        pool.close()
        pool.join()
        return

    def load_agents(self, events):
        agents = pd.DataFrame()
        # removes drivers
        agents =  events[pd.to_numeric(events['person'], errors='coerce').notnull()] 
        dfs = [x for _, x in agents.groupby("person")] #each person in own dataframe
        del agents
        gc.collect()
        return dfs

    def load_vehicles(self, events):
        vehicle_dfs = []
        vehicle_dfs_types = []
        for veh_type in vehicle_types:
            print("\t Grouping",veh_type,":")
            vehicles = pd.DataFrame()
            if veh_type == 'car':
                vehicles = events[pd.to_numeric(events['vehicle'], errors='coerce').notnull()]
            else:
                vehicles = events.loc[events['vehicle'].str.contains(veh_type, case=False)]
                driver_events = events[events['vehicleId'].notnull() & events['vehicleId'].str.contains(veh_type, case=False)]
                driver_events['vehicle'] = driver_events['vehicleId']
                vehicles = vehicles.append(driver_events)


            vehs = [x for _, x in vehicles.groupby("vehicle")]
            vehicle_dfs.extend(vehs)
            vehicle_dfs_types.extend([veh_type]*len(vehs))
            print("\t",veh_type,"# ",len(vehs))
            
        args = [[df,t] for df,t in zip(vehicle_dfs, vehicle_dfs_types)]
        return args

    def clear_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)


    def load_agents_from_population(self,path):
        print("Loading file:", path)
        events = pd.read_csv(path, dtype=events_dtypes).fillna(np.nan)
    
        agent_loads = []

        print("Parsing events:")
        print("\t Grouping agents:")
        dfs = self.load_agents(events)

        print("\t agents # ",len(dfs))
        agent_loads.append(len(dfs))

        events.vehicle = events.vehicle.astype("string")
        
        args = self.load_vehicles(events)
        agent_loads.append(len(args))

        total_agents = sum(agent_loads)
        cpu_available = os.cpu_count()
        #cpu_available = 1

        del events
        gc.collect()
        self.save_vehicles_parallel(args, cpu_available)
        self.save_agents_parallel(dfs, cpu_available)
        return






