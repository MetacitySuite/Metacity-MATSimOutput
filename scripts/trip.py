import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Trip:
    def __init__(self, time, passengers):
        self.locations = []
        self.locations_sec = []
        self.times = []
        self.start = time
        self.passengers = passengers
        self.destination = ''
        self.route_id = -1
        self.line_id = -1
        #self.vehicle_type =

    def get_locations_by_second(self, show_trip = False): 
        new_locations = []
        last_loc = [-1,-1]
        last_time = -1

        for l,t in zip(self.locations, self.times):
            is_close = np.isclose(l, last_loc, rtol=1e-05, atol=1e-08, equal_nan=False)
            if(last_time == -1):
                new_locations.append(l)
            elif (np.isnan(l[0])):
                continue
            elif last_time != -1 and not (is_close[0] and is_close[1]) and  not (t == last_time): # time changed and place changed
                t_diff = int(t - last_time)
                i_x = ((l[0] - last_loc[0])/np.float64(t_diff))
                i_y = ((l[1] - last_loc[1])/np.float64(t_diff))
                # for each second of difference
                new_locations.extend([ [last_loc[0] + t*i_x, last_loc[1]+ t*i_y] for t in range(1,t_diff)])
            elif (is_close[0] and is_close[1]) and  not (t == last_time): # time changed but not place
                new_locations.extend([ [l[0],l[1]] for t in range(int(t - last_time))  ])
            last_loc = l
            last_time = t

        if(show_trip):
            G = nx.DiGraph() 
            last_node = -1
            for loc in new_locations:
                G.add_node(str(loc), x=float(loc[0]), y=float(loc[1]))
                if last_node != -1:
                    G.add_edge(str(last_node),str(loc))
                last_node = loc

            positions = {}
            for idx,node in G.nodes(data=True):
                positions[idx] = [node['x'],node['y']]

            nx.draw_networkx_nodes(G, positions, node_color='r', alpha = 0.1, node_size = 10)
            fig_size=[5,5]
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('equal')
            plt.show()
        
        self.locations_sec = new_locations

    def append_time(self, t):
        self.times.append(t)

    def append_location(self, l):
        self.locations.append(l)

    def set_times(self, times):
        self.times = times

    def set_locations(self, locations):
        self.locations = locations
        self.get_locations_by_second()

    def add_passenger(self, passenger):
        self.passengers.add(int(passenger))

    def remove_passenger(self, passenger):
        self.passengers.remove(int(passenger))