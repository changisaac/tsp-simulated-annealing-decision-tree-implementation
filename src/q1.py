"""
CS 486 Assignment 2 Question 1
Author: Isaac Chang 
Date: Nov 3, 2019
"""

import sys
import math
import matplotlib.pyplot as plt
import random
import time
import pdb

class TSPSimAnneal:

    def __init__(self):
        self.data_dir = '../data/randTSP'

    def main(self):
        lowest_cost = float('inf')

        for j in range(1,2):
            print "36 City Problem"
            print 'Iteration Number: ' + str(j)
            test_file = self.data_dir + '/problem36'
            num_cities, g = self.read_in(test_file)
            
            start_time = time.time()
            final_path, dist_travelled, all_dists = self.sim_anneal_tsp(num_cities, g, 100)

            # plot how cost of solution changes 
            x = range(len(all_dists))
            y = all_dists

            plt.plot(x, y)
            plt.xlabel('Simulated Annealing Iteration Number (unit)') 
            plt.ylabel('Cost of the Solution (unit)') 
            plt.title('Cost of the Solution vs. Simulated Annealing Iteration Number') 
            plt.show() 
            
            print 'final path: ' + str(final_path)
            print 'distance travelled: ' + str(dist_travelled) + ' units'
            print 'time taken: ' + str(time.time() - start_time) + " second(s)" 

            lowest_cost = min(lowest_cost, dist_travelled)
            #self.display_cities(g, final_path)

        print 'lowest Cost: ' + str(lowest_cost) + ' units'
            
    # Simulated annealing solving function
    # param: num_cities (int), g (dict), t (float)
    # return: path (list)
    def sim_anneal_tsp(self, num_cities, g, t):
        all_dists = []
        curr_path = self.calc_init_path(num_cities, g, 'A')
        curr_path_dist = self.total_path_dist(g, curr_path)

        if len(curr_path) <= 2:
            return curr_path
        
        shortest_path = curr_path[:]
        shortest_path_dist = curr_path_dist
        
        num_iter = 0

        while t > 0.01:
            num_iter += 1
            #print 'Iteration Number: ' + str(num_iter)
            # local search using 2-opt to get moveset
            # select ordering in 2-opt randomly
            # omit start and end indices as they cannot be changed
            i = random.randint(1, len(curr_path)-3)
            k = random.randint(i+1, len(curr_path)-2)

            # deep copy here for new path
            new_path = curr_path[:]
            new_path[i:k+1] = reversed(curr_path[i:k+1])

            new_path_dist = self.total_path_dist(g, new_path)
            dist_delta = new_path_dist - curr_path_dist
            
            # decrease t by schedule
            # uncomment to decrease based on exponential
            t = self.decrease_t_exp(0.99999, t)
            #t = self.decrease_t_exp(0.9999, t)
            
            # uncomment to decrease based on linear
            #t = self.decrease_t_lin(0.002, t)

            #print 'Temperature: ' + str(t)

            # if shorter new path, dist_delta is negative
            if dist_delta < 0:
                curr_path = new_path
                curr_path_dist = new_path_dist
                
                if curr_path_dist < shortest_path_dist:
                    shortest_path = curr_path
                    shortest_path_dist = curr_path_dist
            # else dist_delta is positive since new path is longer
            elif t > 0:
                # calc probability and update t
                prob = self.calc_boltzmann_prob(t, dist_delta) 

                #print 'Probability: ' + str(prob)

                # iterate to path based on probability
                if random.random() < prob:
                    curr_path = new_path
                    curr_path_dist = new_path_dist

            all_dists.append(curr_path_dist)

        return shortest_path, shortest_path_dist, all_dists
    
    def calc_boltzmann_prob(self, t, dist_delta):
        # decrease t based on annealing schedules
        
        # linear decrease of schedule of t
        #t = self.decrease_t_lin(-0.5, t)

        # exponential decrease schedule of t 
        #t = self.decrease_t_exp(0.9999, t)
    
        dist_delta = -dist_delta
        prob = math.exp(dist_delta / t)

        return prob
   
    def decrease_t_lin(self, delta, t):
        t = t - delta
        return t

    def decrease_t_exp(self, alpha, t):
        t = t * alpha
        return t

    # Calculate non-optimal initial path
    # param: num_cities (int), g (dict), s_city_name (str)
    # return: path (list)
    def calc_init_path(self, num_cities, g, s_city_name):
        # generate initial path using next closest city
        path = [s_city_name]
        
        # since maximum cities to iterate through  is n
        # and each iteration reqirues iteration through n cities again
        # time complexity to generate inital path is O(n^2)

        while len(path) <  len(g.keys()):
            min_dist = sys.float_info.max
            next_city = None

            for city in g.keys():
                # out of the remaining untravelled cities, find closest
                if city not in path:
                    dist = self.eucl_dist(g[path[-1]], g[city]) 
                   
                    if dist <=  min_dist:
                        min_dist = dist
                        next_city = city

            path.append(next_city)
                   
        # tag on return to start city
        path.append(s_city_name)

        return path
        
    # Helper Classes -----
    
    # A struct class to more easily store city coordinates
    class City:

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __str__(self):
            return "(X: {0}, Y: {1})".format(self.x, self.y)
    
    # Helper Functions -----

    def eucl_dist(self, c1, c2):
        x_delta = c1.x - c2.x
        y_delta = c1.y - c2.y

        dist = math.sqrt(x_delta ** 2 + y_delta ** 2)

        return dist

    def total_path_dist(self, g, path):
        dist = 0

        for i in range(len(path)-1):
            dist += self.eucl_dist(g[path[i]], g[path[i+1]])

        return dist

    # Setup Functions -----

    def read_in(self, file_name):
        f = open(file_name)
        lines = list(f)

        g = dict()

        num_cities = int(lines[0])

        for i in range(1, len(lines)):
            city, x, y = lines[i].split()
            g[city] = self.City(int(x), int(y))

        return num_cities, g

    def display_cities(self, g, path):
        x_coords = []
        y_coords = []

        for point in path:
            x_coords.append(g[point].x)
            y_coords.append(g[point].y)

        fig, graph = plt.subplots()
        graph.scatter(x_coords, y_coords)
        graph.plot(x_coords, y_coords)

        for i, label in enumerate(path):
            graph.annotate(label, (x_coords[i], y_coords[i]))

        plt.show()

if __name__ == '__main__':
    sol = TSPSimAnneal()
    sol.main()
