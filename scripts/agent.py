import base64
import numpy as np
import pandas as pd
import geopandas as gpd
from trip import Trip
from shapely.geometry import MultiPoint

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
                    trip.append_location(B)
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
                trip.append_location(B)

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
        

    def extract_trips_cars(self, verbal=False):
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
                trip.append_location(B)
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


    def prepare_geotrips(self):
        start_times = []
        passengers = []
        geometries = []
        #times = []
        route_ids = []
        line_ids = []

        #print("prepping geotrips")
        for trip in self.trips:
            start_times.append(int(trip.start))
            route_ids.append(trip.route_id) # id routy str
            line_ids.append(trip.line_id) # id linky str
            passengers.append(npint32_to_buffer(np.array(list(trip.passengers))))
            #times.append(npint32_to_buffer(np.array(trip.times)))
            geometries.append(MultiPoint([(a[0],a[1]) for a in trip.locations_sec]))

        trips_id = self.id
        if(self.type == 'car'):
            trips_id = "veh_"+str(self.id)+"_car"

        agent_geotrips = gpd.GeoDataFrame(data={
            'start': start_times,
            'passengers':passengers,
            #'times':times,
            'geometry': geometries,
            'id': trips_id,
            'veh_type': self.type,
            'route_id': route_ids,
            'line_id' : line_ids,
            't_series': True})

        self.geotrips = agent_geotrips
        #print("# of geotrips:", len(self.geotrips))
