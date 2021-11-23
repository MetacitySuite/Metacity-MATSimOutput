import base64
import numpy as np
import pandas as pd
import geopandas as gpd
from trip import Trip
from shapely.geometry import MultiPoint
import networkx as nx
import matplotlib.pyplot as plt

def npint32_to_buffer(data):
    return base64.b64encode(data.astype(np.int32)).decode('utf-8')

def npfloat64_to_string(data):
    data = np.array(data, dtype=np.float64)
    return base64.b64encode(data).decode('utf-8')

# import
def base64_to_type(b64data, type):
    bdata = base64.b64decode(b64data)
    data = np.frombuffer(bdata, dtype=type)
    return data

def base64_to_int32(b64data):
    return base64_to_type(b64data, np.int32)


class Agent:
    def __init__(self,agent_type,agent_id):
        self.type =  agent_type
        self.trips = []
        self.geotrips = gpd.GeoDataFrame()
        self.id = agent_id
        self.events = pd.DataFrame()

    def set_events(self, events):
        events['type'] = pd.Categorical(events['type'], 
                ["actstart","actend","VehicleArrivesAtFacility", "PersonLeavesVehicle","PersonEntersVehicle","VehicleDepartsAtFacility","entered_link", "TransitDriverStarts"])
        ##sort vehicle by time and type
        events = events.sort_values(["time","type"],kind="stable")
        self.events = events
        #print("# of events in agent events:", self.events.shape)

    def extract_trips(self, verbal=False):
        pass
        

    def prepare_geotrips(self, format):
        trips_id = self.id
        if(self.type == 'car'): # create consistent id for all vehicles
            trips_id = "veh_"+str(self.id)+"_car"

        if(format == 'shp'):
            start_times = []
            passengers = []
            geometries = []
            route_ids = []
            line_ids = []

            for trip in self.trips:
                start_times.append(np.int32(trip.start))
                if(self.type not in ["agent", "car"]):
                    route_ids.append(trip.route_id) # id routy str
                    line_ids.append(trip.line_id) # id linky str
                passengers.append(npint32_to_buffer(np.array(list(trip.passengers))))
                #passengers.append(list(map(int, list(trip.passengers))))
                geometries.append(MultiPoint([(a[0],a[1]) for a in trip.locations_sec]))
                #geometries.append(([(a[0],a[1]) for a in trip.locations_sec]))

            agent_geotrips = gpd.GeoDataFrame(data={
                'start': start_times,
                'passengers':passengers,
                'geometry': geometries,
                'id': trips_id,
                'veh_type': self.type,
                'metatype': "time_series"})
            
            if(self.type not in ["agent", "car"]):
                agent_geotrips["route_id"] = route_ids
                agent_geotrips["line_id"] = line_ids


        elif(format == 'json'):
            agent_geotrips = []
            
            for trip in self.trips:
                meta = {}
                meta["passengers"] = list(map(int, list(trip.passengers)))
                meta["start"] = int(trip.start)
                meta["id"] = trips_id
                if(self.type == "agent"):
                    meta["veh_type"] = trip.vehicle_id.split('_')[-1]
                    meta["vehicle_id"] = trip.vehicle_id
                else:
                    meta["veh_type"] = self.type
                    meta["vehicle_id"] = trips_id

                if(self.type not in ["agent", "car"]):
                    meta['route_id'] = trip.route_id
                    meta['line_id'] = trip.line_id

                points = [np.float64(b) for a in trip.locations_sec for b in a]
                geometry = npfloat64_to_string(points)
                geotrip = {
                'meta' : meta,
                'geometry': geometry
                }

                agent_geotrips.append(geotrip)

        self.geotrips = agent_geotrips
  

class MHD(Agent):
    def __init__(self, agent_type,id):
        super().__init__(agent_type,id)

    def extract_trips(self, verbal=False):
        self.trips =[]
        in_station = False

        old_passengers = set()
        trip = Trip(-1, old_passengers)

        if(self.type != 'car' and self.type != "agent"):
            if("transitRoute" not in self.events.columns):
                print("No routes per transit agent", self.id)
                #display(self.events)
                trip_route = "Unknown"
                trip_line = "Unknown"

            elif(len(self.events.transitRoute.unique())<2):
                print("No valid routes per transit agent",self.events.transitRoute.unique(), self.events.transitLine.unique(), self.id)
                #display(self.events)
                trip_route = "Unknown"
                trip_line = "Unknown"

            else:
                trip_route = self.events.transitRoute.unique()[1]
                trip_line = self.events.transitLine.unique()[1]
                if(len(self.events.transitRoute.unique())>2):
                    print("more routes per transit agent",self.events.transitRoute.unique())

        
        for e, row in self.events.iterrows():
            A = row.coords_from
            B = row.coords_to
            time = row.time

            #end and start trip
            if not in_station and row.type == "VehicleArrivesAtFacility":
                in_station = True
                #save finished Trip
                if (trip.start > -1):
                    trip.append_time(time)
                    trip.append_location(B) #add end
                    #trip.destination = str(row.facility) unused
                    trip.get_locations_by_second()
                    self.trips.append(trip)
                    old_passengers = trip.passengers

                #start new Trip
                trip = Trip(time, old_passengers)
                if( self.type != "car" and self.type != "agent"):
                    trip.route_id = trip_route
                    trip.line_id = trip_line
                trip.append_time(time)
                trip.append_location(A) # add start

            elif row.type == "PersonEntersVehicle":
                #check if passenger leaves vehicle
                two_interactions = self.events.iloc[np.where(self.events.person_id == row.person_id)].shape[0] > 1
                if two_interactions and str(row.person_id).isnumeric():
                    trip.add_passenger(row.person_id)
                elif str(row.person_id).isnumeric():
                    print("Passenger:", row.person_id,"does not leave vehicle.")

            elif row.type == "PersonLeavesVehicle":
                if(str(row.person_id).isnumeric()):
                    try:
                        trip.remove_passenger(row.person_id)
                    except KeyError:
                        print("\tCould not remove passenger",row.person_id,"from", self.id, trip.passengers)

            #start trip
            elif in_station and (row.type == "VehicleDepartsAtFacility"):
                in_station = False

            if not np.isnan(A[0]):
                trip.append_time(time)
                trip.append_location(A)

            if(not in_station and ((row.type == "PersonEntersVehicle") or (row.type == "PersonLeavesVehicle")) and row.person_id.isnumeric()):
                print("Vehicle is not in station and there are changes in passenger list",
                row.type, row.person_id, row.time)


class Car(Agent):
    def __init__(self, agent_type,id):
        super().__init__("car",id)

    def extract_trips(self, verbal=False):
        self.trips = []

        old_passengers = set()
        trip = Trip(-1, old_passengers)

        for e, row in self.events.iterrows():
            A = row.coords_from
            B = row.coords_to
            time = row.time

            #end and start trip  
            if row.type == "PersonEntersVehicle": #1 passenger per car
                #start new Trip
                #print("Vehicle arrived at facility:", row.facility)
                trip = Trip(time, old_passengers)
                trip.append_time(time)
                trip.append_location(A)
                #check if passenger leaves vehicle
                two_interactions = self.events.iloc[np.where(self.events.person_id == row.person_id)].shape[0] > 1
                if two_interactions and str(row.person_id).isnumeric():
                    #print("Adding passenger:", row.person_id)
                    trip.add_passenger(row.person_id)
                elif str(row.person_id).isnumeric():
                    print("Passenger:", row.person_id,"does not leave vehicle.")

            elif row.type == "PersonLeavesVehicle":
                if(str(row.person_id).isnumeric()):
                    #print("Removing passenger:", row.person_id)
                    trip.append_time(time)
                    trip.append_location(B)
                    #trip.destination = str(row.facility) unused
                    trip.get_locations_by_second()
                    self.trips.append(trip)
                    old_passengers = trip.passengers

            if not np.isnan(A[0]):
                trip.append_time(time)
                trip.append_location(A)

class Human(Agent):
    def __init__(self, agent_type,id):
        super().__init__("agent",id)
        self.home = []
        self.home_from = -1
        self.home_to = -1
        self.work = []
        self.work_from = -1
        self.work_to = -1


    def get_facility_coords(self, facility_type="work"):
        #facility_types = ['work','home']
        home_coords = [-1,-1]
        time_from = 0.0; time_to = 0.0
        facility_events = self.events.iloc[np.where(self.events['actType'] == facility_type)]
        home_coords = [-facility_events.coords_x.unique()[0],-facility_events.coords_y.unique()[0]] 

        if(facility_type in ["home","work"] and facility_events.shape[0] >= 2):
            time_from = facility_events.iloc[np.where(facility_events.type == "actstart")].time.unique()[0]
            time_to = facility_events.iloc[np.where(facility_events.type == "actend")].time.unique()[0] 
            return home_coords, time_from, time_to 
        else:
            #print("Person did not arrive safely.")
            return home_coords, -1, -1
    
    def extract_trips(self, verbal=False):
        #self.trips = []
        trip = Trip(-1, set())
        in_trip = False
        in_mhd = False
        in_car = False
        
        if("stuckAndAbort" in self.events.type.unique()):
            print("Agent",self.id, "is stuck.")
            return

        if(verbal):
            G = nx.DiGraph() 
            colors = ['g','k','b','orange','violet','lime','yellow','slateblue','tomato','indigo','olive']
            act_types = ["home","work","pt interaction", np.nan]
            special = ["home","work"]
            act_sizes = [300,300,100,5]
            act_colors = ["g","r","orange","b"]
            vehicles = []
            acts = set()


        self.home, self.home_from, self.home_to = self.get_facility_coords("home")
        if( self.home_from < 0):
            print("Agent",self.id, "did not arrive home.")
            return

        self.work, self.work_from, self.work_to = self.get_facility_coords("work")
        if( self.work_from < 0):
            print("Agent",self.id, "did not arrive at work.")
            return


        for e, row in self.events.iterrows():
            #print(act, A, B)
            A = row.coords_from
            B = row.coords_to
            v = row.vehicle_id
            act = row.actType
            time = row.time
            event_type = row.type

            if not in_trip and ((event_type == 'PersonEntersVehicle') or (event_type == "waitingForPt")): 
                # add waiting for pt times
                in_trip = True
                trip = Trip(time, set([str(self.id)]))
                trip.start = time
                trip.append_time(time)
                trip.append_location(A)
                
            if in_trip and (event_type == 'PersonEntersVehicle' and not v.isnumeric()):
                in_mhd = True
                trip.vehicle_id = v
            elif in_trip and (event_type == 'PersonEntersVehicle' and v.isnumeric()):
                in_car = True
                trip.vehicle_id = "veh_"+v+"_car"
                    
            if in_trip and not np.isnan(A[0]):
                trip.append_time(time)
                trip.append_location(A)

            if(event_type == 'PersonLeavesVehicle' and in_trip and in_car) or (event_type == 'PersonLeavesVehicle' and in_trip and in_mhd):
                #print("Location B", B)
                trip.append_location(B) #probably always nan 
                trip.append_time(time)
                trip.get_locations_by_second()
                self.trips.append(trip)
                in_trip = False
                in_mhd = False
                in_car = False

            if(verbal):
                acts.add(act)
                if act in special or (not str(A) in G and not np.isnan(A[0])):
                    G.add_node(str(A), x=float(A[0]), y=float(A[1]), act_type=act_sizes[act_types.index(act)], act_color=act_colors[act_types.index(act)])
                if act in special or (not str(B) in G and not np.isnan(B[0])):
                    G.add_node(str(B), x=float(B[0]), y=float(B[1]), act_type=act_sizes[act_types.index(act)], act_color=act_colors[act_types.index(act)])

                if (A[0] != B[0]) and (A[1] != B[1]) and not np.isnan(A[0]):
                    if(v not in vehicles):
                        vehicles.append(v)
                    #if(np.isnan(v) and legMode=="walk"): last_to -> walk_from
                    G.add_edge(str(A),str(B), mode=colors[vehicles.index(v)])

        if(verbal):
            print(vehicles)
            #print(acts)
            positions = {}
            for idx,node in G.nodes(data=True):
                #print(idx, node)
                positions[idx] = [node['x'],node['y']]

            nx.draw_networkx_nodes(G, positions, node_color=[u['act_color'] for i,u in G.nodes(data=True)], alpha = 0.5, node_size = [u['act_type'] for i,u in G.nodes(data=True)])
            nx.draw_networkx_edges(G, positions, edge_color=[G[u][v]['mode'] for u,v in G.edges()],alpha=0.3, arrows = True)
            fig_size=[10,10]
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('equal')

