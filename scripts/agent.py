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

# import
def base64_to_type(b64data, type):
    bdata = base64.b64decode(b64data)
    data = np.frombuffer(bdata, dtype=type)
    return data

def base64_to_int32(b64data):
    return base64_to_type(b64data, np.int32)


class Agent:
    def __init__(self,agent_type,id):
        self.type =  agent_type
        self.trips = []
        self.geotrips = gpd.GeoDataFrame()
        self.id = id
        self.events = pd.DataFrame()

    def set_events(self, events):
        events['type'] = pd.Categorical(events['type'], 
                ["VehicleArrivesAtFacility", "PersonLeavesVehicle","PersonEntersVehicle","VehicleDepartsAtFacility"])
        ##sort vehicle by time and type
        events = events.sort_values(["time","type"],kind="stable")
        self.events = events
        #print("# of events in agent events:", self.events.shape)

    def extract_trips(self, verbal=False):
        pass
        

    def prepare_geotrips(self):
        pass
  

class MHD(Agent):
    def __init__(self, agent_type,id):
        super().__init__(agent_type,id)

        def prepare_geotrips(self):
            start_times = []
            passengers = []
            geometries = []
            route_ids = []
            line_ids = []

            for trip in self.trips:
                start_times.append(int(trip.start))
                route_ids.append(trip.route_id) # id routy str
                line_ids.append(trip.line_id) # id linky str
                passengers.append(npint32_to_buffer(np.array(list(trip.passengers))))
                geometries.append(MultiPoint([(a[0],a[1]) for a in trip.locations_sec]))

            trips_id = self.id
            agent_geotrips = gpd.GeoDataFrame(data={
                'start': start_times,
                'passengers':passengers,
                'geometry': geometries,
                'id': trips_id,
                'veh_type': self.type,
                'route_id': route_ids,
                'line_id' : line_ids,
                'metatype': "time_series"})
            self.geotrips = agent_geotrips

    def extract_trips(self, verbal=False):
        self.trips =[]
        in_station = False

        old_passengers = set()
        trip = Trip(-1, old_passengers)

        if(self.type != 'car' and self.type != "agent"):
            trip_route = self.events.transitRoute.unique()[1]
            #print(trip_route)
            trip_line = self.events.transitLine.unique()[1]
            #print(trip_line)
            if(len(self.events.transitRoute.unique())>2):
                print("more routes per agent",self.events.transitRoute.unique())
        #print(self.events.transitRoute.unique())
        
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
                #print("Vehicle arrived at facility:", row.facility)
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
                    #print("Adding passenger:", row.person_id)
                    trip.add_passenger(row.person_id)
                elif str(row.person_id).isnumeric():
                    print("Passenger:", row.person_id,"does not leave vehicle.")

            elif row.type == "PersonLeavesVehicle":
                if(str(row.person_id).isnumeric()):
                    #print("Removing passenger:", row.person_id)
                    trip.remove_passenger(row.person_id)

            #start trip
            elif in_station and (row.type == "VehicleDepartsAtFacility"):
                #print("\tdeparts")
                in_station = False


            if not np.isnan(A[0]):
                trip.append_time(time)
                trip.append_location(A)

            if(not in_station and ((row.type == "PersonEntersVehicle") or (row.type == "PersonLeavesVehicle")) and row.person_id.isnumeric()):
                print("Vehicle is not in station and there are changes in passenger list",
                row.type, row.person_id, row.time)

        #print("# of extracted trips for agent",self.id,":", len(self.trips))

class Car(Agent):
    def __init__(self, agent_type,id):
        super().__init__("car",id)

        def prepare_geotrips(self):
            start_times = []
            passengers = []
            geometries = []

            for trip in self.trips:
                start_times.append(int(trip.start))
                passengers.append(npint32_to_buffer(np.array(list(trip.passengers))))
                geometries.append(MultiPoint([(a[0],a[1]) for a in trip.locations_sec]))

            trips_id = self.id
            if(self.type == 'car'): # create consistent id for all vehicles
                trips_id = "veh_"+str(self.id)+"_car"

            agent_geotrips = gpd.GeoDataFrame(data={
                'start': start_times,
                'passengers':passengers,
                'geometry': geometries,
                'id': trips_id,
                'veh_type': self.type,
                'metatype': "time_series"})
            self.geotrips = agent_geotrips

    def extract_trips(self, verbal=False):
        self.trips = []

        old_passengers = set()
        trip = Trip(-1, old_passengers)

        for e, row in self.events.iterrows():
            A = row.coords_from
            B = row.coords_to
            time = row.time

            #end and start trip  
            if row.type == "PersonEntersVehicle": #1 passanger per car
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

    def prepare_geotrips(self):
        start_times = []
        passengers = []
        geometries = []
        vehicle_ids = []

        for trip in self.trips:
            start_times.append(int(trip.start))
            passengers.append(npint32_to_buffer(np.array(list(trip.passengers))))
            geometries.append(MultiPoint([(a[0],a[1]) for a in trip.locations_sec]))
            vehicle_ids.append(trip.vehicle_id)
        trips_id = self.id

        agent_geotrips = gpd.GeoDataFrame(data={
            'start': start_times,
            'passengers':passengers,
            'geometry': geometries,
            'id': trips_id,
            'veh_type': self.type,
            'vehicle_id': vehicle_ids,
            'metatype': "time_series"})
        self.geotrips = agent_geotrips


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
        self.trips = []
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

            if not in_trip and ((event_type == 'PersonEntersVehicle' and v.isnumeric()) or (event_type == "waitingForPt")): 
                # add waiting for pt times
                in_trip = True
                trip = Trip(time, set(self.id))
                trip.start = time
                if(v.isnumeric()):
                    trip.vehicle_id = "veh_"+v+"_car"
                else:
                    trip.vehicle_id = v


            if in_trip and (event_type == 'PersonEntersVehicle' and not v.isnumeric()):
                in_mhd = True
            elif in_trip and (event_type == 'PersonEntersVehicle' and v.isnumeric()):
                in_car = True

            #if(event_type == 'arrival'):
            if(event_type == 'PersonLeavesVehicle' and in_trip and in_car) or (event_type == 'PersonLeavesVehicle' and in_trip and in_mhd):
                trip.append_location(B)
                trip.append_time(time)
                self.trips.append(trip)
                in_trip = False
                in_mhd = False
                in_car = False

            if in_trip and not np.isnan(A[0]):
                trip.append_time(time)
                trip.append_location(A)

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

