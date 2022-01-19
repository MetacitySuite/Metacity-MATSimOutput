import pandas as pd
import os
import numpy as np
import pandas as pd
import os
import shutil
import json
import warnings
from tqdm import tqdm
import gc
from multiprocessing import Pool

#pd.set_option('display.max_columns', None)
#warnings.filterwarnings('ignore')

EVENTS = "./output/events/"
AGENTS = "./output/agents/"

vehicle_types = [ "car","bike","ferry","bus","funicular","subway", "tram"]
#vehicle_types = ["car"]


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

new_columns_pt = ["events_id","time","vehicle","type","link","person_id","delay","facility","networkMode"
                ,"actType","legMode","vehicle_id","coords_x","coords_y","relativePosition","transitLine",
                "transitRoute","departure","atStop","destinationStop"]

new_columns_car = ["events_id","time","vehicle","type","link","person_id","delay","facility","networkMode"
                ,"actType","legMode","vehicle_id","coords_x","coords_y","relativePosition"]

def return_link(facility, link):
    if (isinstance(facility, str)):
        return facility.split(':')[-1]
    return link



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

    if(vehicle_type != "car" or vehicle_type != "bike"):
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

        chunk = Chunk(agent_id)
        chunk.events = []
        chunk.events = agent.to_dict('records')
        chunk.save_chunk(self.agents_path+"/agent","/"+str(agent_id)+".json")
        del chunk
        gc.collect()
        return


    def process_vehicle(self, args):
        vehicle_df, vehicle_type = args
        vehicle = vehicle_df.sort_values("time") #prepped for linking
        
        if(vehicle_type == "car"):
            vehicle_id = int(vehicle.vehicle.unique()[0])
        else:
            vehicle_id = vehicle.vehicle.unique()[0]
    
        chunk = Chunk(vehicle_id)
        chunk.events = []
        chunk.events = vehicle.to_dict('records')
        chunk.save_chunk(self.agents_path+"/"+vehicle_type,"/"+str(vehicle_id)+".json")
        del chunk
        gc.collect()
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
            pool.map(self.process_vehicle, tqdm(args))

        pool.close()
        pool.join()
        del args
        return

    def save_agents_parallel(self, persons, cpus):
        print("Number of agents in loaded chunk:", len(persons))
        if(len(persons) < 1):
            return
        print("processing:")

        with Pool(cpus) as pool:
            pool.map(self.process_agent, tqdm(persons))

        pool.close()
        pool.join()
        del persons
        return

    def load_agents(self, events):
        agents = pd.DataFrame()
        # remove drivers
        agents =  events[pd.to_numeric(events['person'], errors='coerce').notnull()] 
        #rename columns and drop unused
        agents = agents.rename(columns={0:"events_id","vehicle":"vehicle_id","x":"coords_x","y":"coords_y"})
        new_columns = ["events_id","time","person","type","link","delay","actType","legMode","vehicle_id","coords_x","coords_y"]
        old_columns = agents.columns
        agents = agents.drop((set(old_columns) - set(new_columns)),axis=1)
        dfs = [x for _, x in agents.groupby("person", sort=False)] #each person in own dataframe
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
            elif veh_type == 'bike':
                vehicles = events.loc[events['vehicle'].str.contains("_bike", case=False)]
            else:
                vehicles = events.loc[events['vehicle'].str.contains(veh_type, case=False)]
                driver_events = events[events['vehicleId'].notnull() & events['vehicleId'].str.contains(veh_type, case=False)]
                driver_events.loc[:, "vehicle"] = driver_events.vehicleId.values
                driver_events.drop(["vehicleId"], axis=1, inplace=True)
                vehicles = vehicles.append(driver_events, ignore_index=True, verify_integrity=True)


            if(veh_type == "car" or veh_type == "bike"):
                vehicles = vehicles.rename(columns={0:"events_id","person":"person_id",
                                    "x":"coords_x","y":"coords_y"})
                old_columns = vehicles.columns
                vehicles = vehicles.drop((set(old_columns) - set(new_columns_car)),axis=1)
            else:
                vehicles = vehicles.rename(columns={0:"events_id","person":"person_id",
                                    "x":"coords_x","y":"coords_y","transitLineId":"transitLine",
                                    "transitRouteId":"transitRoute","departureId":"departure"})
                old_columns = vehicles.columns
                vehicles.drop((set(old_columns) - set(new_columns_pt)),axis=1)
                
                vehicles.loc[:,"link"] = vehicles.apply(lambda row: return_link(row.facility, row.link),axis=1)

            vehs = [x for _, x in vehicles.groupby("vehicle", sort=False)]
            v_len = len(vehs)
            vehicle_dfs.extend(vehs)
            vehicle_dfs_types.extend([veh_type]*v_len)
            print("\t",veh_type,"# ",v_len)
            
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

        print("\t people # ",len(dfs))
        agent_loads.append(len(dfs))

        events.vehicle = events.vehicle.astype("string")
        
        args = self.load_vehicles(events)
        agent_loads.append(len(args))

        cpu_available = os.cpu_count()

        del events
        gc.collect()
    
        self.save_vehicles_parallel(args, cpu_available)

        self.save_agents_parallel(dfs, cpu_available)
        
        return






