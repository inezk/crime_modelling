# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:43:09 2018

@author: inezk
"""

import numpy as np

def mu_nu(x, y):
    rate = (5.71/(2 * np.pi * 4.5 **2)) * np.exp(-x **2 / (2 * 4.5 **2)) * np.exp(-y ** 2/ (2 * 4.5 **2))
    return rate

def g(x, y, t):
    rate = 0.2 * 0.1 * np.exp(-0.1 * t) *np.exp(-x **2 /(2 * 0.01 ** 2)) * np.exp(-y ** 2/ (2 * 0.1 ** 2))
    return rate

def ParentPP(x, y, time):
    denom = time * (5.71/4) #half mu bar
    num_events = np.random.poisson(lam = denom)
    events = []
    while len(events) <= num_events:
        x_coord = np.random.uniform(-x,x,size = 1000).reshape(1000,1)
        y_coord = np.random.uniform(-y,y,size = 1000).reshape(1000,1)
        t_coord = np.random.random_integers(low = 0,high = time, size = 1000).reshape(1000,1)
        num = mu_nu(x_coord, y_coord)
        random_num = np.random.uniform(size = 1000).reshape(1000,1)
        ind = np.less_equal(random_num, num/denom)
        ind = np.where(ind == True)
        if len(ind[0]) > 0:
            num_ind = len(x_coord[ind])
            events += [np.concatenate((x_coord[ind].reshape(num_ind, 1), 
                                       y_coord[ind].reshape(num_ind, 1), 
                                       t_coord[ind].reshape(num_ind, 1)), axis = 1)]
        print("# background",len(events))
    events = np.concatenate(events, axis = 0)[0:num_events,:]
    np.savetxt("background_small.csv", events, delimiter = ", ", fmt = "%5s")
    return events

def ChildPP(x, y, t, denom, grid_x, grid_y, time):
    if t >= time: return np.array([])
    num_events = np.random.poisson(lam = denom)
    events = []
    while len(events) <= num_events and denom > 0:
        x_coord = np.random.uniform(-x,x,size = 1000).reshape(1000,1)
        y_coord = np.random.uniform(-y,y,size = 1000).reshape(1000,1)
        t_coord = np.random.random_integers(low = t,high = time, size = 1000).reshape(1000,1)
        num = g(x -x_coord, y - y_coord, t- t_coord)
        random_num = np.random.uniform(size = 1000).reshape(1000,1)
        ind = np.less_equal(random_num, num/denom)
        ind = np.where(ind == True)
        if len(ind[0]) > 0:
            num_ind = len(x_coord[ind])
            events += [np.concatenate((x_coord[ind].reshape(num_ind, 1), 
                                       y_coord[ind].reshape(num_ind, 1), 
                                       t_coord[ind].reshape(num_ind, 1)), axis = 1)]
    if len(events) > 0: events = np.concatenate(events, axis = 0)[0:num_events,:]
    return events

def generate_points(grid_x, grid_y, time, filename):
    parents = ParentPP(grid_x, grid_y, time)
    #parents = np.genfromtxt("background_small.csv", delimiter = ",")
    child_denom = 0.2
    num = 0
    children = []
    num += 1
    for i in range(0, len(parents)):
        print(i)
        events = ChildPP(parents[i, 0], parents[i, 1], parents[i, 2], child_denom,
                         grid_x, grid_y, time)
        if len(events) != 0: children += [ events]
    children = np.concatenate(children, axis = 0)
    print(len(children))
    all_events = np.concatenate((parents, children), axis = 0)
    while len(children) > 0:
        print("num",num)
        new_children = []
        for i in range(0, len(children)):
            events = ChildPP(children[i, 0], children[i, 1], children[i, 2], child_denom,
                             grid_x, grid_y, time)
            if len(events) > 0:
                 new_children += [events]  
        if len(new_children) > 0: 
            children = np.concatenate(new_children, axis = 0)
            all_events = np.concatenate((all_events, children), axis = 0)
        else: children = []
        num += 1
        print("#children",len(children))
    print(len(all_events))
    #factor = int(0.2 * len(all_events))
    #all_events = all_events[:len(all_events) - factor,:]
    #all_events = all_events[factor:,:]
    np.random.shuffle(all_events)
    add_col = np.zeros((len(all_events), 2))
    add_col[:,1] = list(range(0, len(all_events)))
    all_events = np.concatenate((all_events, add_col), axis = 1)
    np.savetxt(filename, all_events, delimiter = ", ", fmt = "%5s", 
               header = "XCOORD,YCOORD,Time,Type,ID")
    return all_events

generate_points(100, 100, 1000, "sample_test_small.csv")  
#print(ChildPP(10, 10, 5, 0.2, 20, 20, 100) )