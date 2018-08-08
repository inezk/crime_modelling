<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Hawkes Process as per Mohler et al. (2011) - class file
Created on Wed Jul 11 11:20:59 2018

@author: inezk
"""
import numpy as np
from predictor_class import Predictor
from sklearn.neighbors.kde import KernelDensity
from scipy.spatial.distance import pdist, squareform
import copy
import csv

#note - (t, x,y) sorted by time

class hawkes_process(Predictor):
    def __init__(self, data, filename, moving_window = 4, fixed_bandwidth = True, bandwidth = 500, u_k = 15, 
                 v_k = 100, start_index = 104, end_index = 260, crime_types = []):
        Predictor.__init__(self, data, filename)
        self.model = ""
        self.predictions = self.predict(int(moving_window), int(start_index), int(end_index),
                                         fixed_bandwidth, int(bandwidth), int(u_k), int(v_k), crime_types)

    #getting v,u,g
    def train(self, background, offspring, fixed_bandwidth, bandwidth, u_k, v_k): #v, u , g
        if fixed_bandwidth:
            v = KernelDensity(bandwidth)
            u = KernelDensity(bandwidth)
            g = KernelDensity(bandwidth)
            
            v.fit(background[:, 0].reshape(-1, 1))
            u.fit(background[:, 1:])
            g.fit(offspring)
        else: #scale so variance is 1 for offspring and background data
            offspring_scaled = offspring/(np.var(offspring,axis = 1).reshape(len(offspring), 1))
            background_scaled = background/(np.var(background,axis = 1).reshape(len(background), 1))
            #get D, matrix of nearest neighbor distances to each data point
            offspring_scaled = squareform(pdist(offspring_scaled))
            background_scaled_v = squareform(pdist(background_scaled[:,0].reshape(len(background), 1)))
            background_scaled_u = squareform(pdist(background_scaled[:,1:]))
            offspring_scaled.sort(axis=1)
            background_scaled_v.sort(axis=1)
            background_scaled_u.sort(axis=1)
            #getting Di, the ith nearest neighbor distance
            Du_i = background_scaled_u[:, u_k]
            Dv_i = background_scaled_v[:, v_k]
            Dg_i = offspring_scaled[:, u_k].reshape(len(offspring),1)
            
            off_sig = np.std(offspring,axis=0, dtype=np.float64) #(t,x,y)
            back_sig = np.std(background, axis = 0, dtype = np.float64) #(t,x,y)
            
            def g(data):  #g(t,x,y)
                n_offspring = offspring.shape[0]
                n_cases = data.shape[0]
                n_dim = data.shape[1]
                #having all entries in matrix (n_cases, n_offspring, n_dim)
                const = (1/(np.prod(off_sig) * (2 * np.pi) ** (3/2) * Dg_i **3))
                const_tiled = np.tile(const, (n_cases, 1, 1)).reshape(n_cases,n_offspring)
                data_tiled = np.tile(data.reshape(n_cases, 1, n_dim), (1,n_offspring,1))
                exponent_denoms = np.tile((2 * (off_sig**2).reshape(1, n_dim) * Dg_i **2).reshape(1, n_offspring, n_dim), 
                                          (n_cases, 1, 1))
                exponents = (-((data_tiled - offspring)**2)/exponent_denoms).sum(axis=(2)).astype(float)
                final_g = (const_tiled*np.exp(exponents)).mean(axis=1)
                return final_g
            
            def v(t):
                const = 1/(back_sig[0] * (2 * np.pi) ** (1/2) * Dv_i)
                exp_t = -(t - background[:,0]) ** 2/(2 * back_sig[0] ** 2 * Dv_i)
                total = const * np.exp(exp_t.astype(float))
                final_v = np.sum(total)/(len(background))
                return final_v
            
            def u(x, y):
                const = 1/(back_sig[1] * back_sig[2] * (2 * np.pi) * Du_i ** 2)
                exp_x = -(x - background[:,1]) ** 2/(2 * back_sig[1] ** 2 * Du_i)
                exp_y = -(y - background[:,2]) ** 2/(2 * back_sig[2] ** 2 * Du_i)
                total = const * np.exp(exp_x.astype(float) + exp_y.astype(float))
                final_u = np.sum(total)/(len(background))
                return(final_u)
                
        return (v, u, g)
    
    def get_instances(self, frame): #converting from counts to instances 
        date_range = frame
        instances = np.sum(date_range[:,:,2:], axis = (2,1), dtype = np.int64)
        xtrain = np.repeat(self.SDS.coords, instances, axis = 0)
        time = np.repeat(date_range[:,:,1], instances, axis = 0)
        time_intsances = np.sum(date_range[:,:,2:], axis = 2, dtype = np.int64)
        time = time[:,np.where(time_intsances > 0)[1]][0]
        whole_frame = np.concatenate((time.reshape(len(time), 1), xtrain), axis = 1)
        whole_frame = whole_frame[whole_frame[:,0].argsort()] 
        return whole_frame
    
    #setting up p_matrix with random probabilities for events j (col)< i (row)
    def initialize_p(self, data):
        p = np.zeros((len(data), len(data)), dtype= float)
        ind_matrix = np.zeros((len(p), 1)) #matrix for storing indices t_k < t
        for col in range(0, len(p)):
            ind = np.where(data[:,0] < data[col,0])[0] 
            if len(ind) > 0: 
                ind = ind[len(ind) - 1] + 1
                ind_matrix[col] = ind
                p[0:ind, col] = np.random.random(ind)
                p[col, col] = 0
                p[:, col] = p[:, col]/(np.sum(p[:,col])* 2)
                p[col, col] = 0.5
            else: 
                p[col, col] = 1
                ind_matrix[col] = -1 #for when no events appear before it
        return (p, ind_matrix)
    
    def predict(self, moving_window, start_index, end_index, fixed_bandwidth, bandwidth,
                u_k, v_k, crime_types):
        if len(crime_types) > 0:
            self.data = self.data[:,:, [0, 1] + crime_types]
        n_space = len(self.data)
        results = np.zeros(((end_index - start_index) * n_space, 5), dtype = object)
        #these lines are for testing purposes - will eventually be removed
        if fixed_bandwidth: file_obj = open("sample_results_true.csv", "w") 
        else: file_obj = open("sample_results_false.csv", "w")
        file = csv.writer(file_obj)
        row_num = 0
        #initializing fields
        counts = self.data
        list_l2_norm = []
        num_background = []
        for i in range(start_index, end_index): 
            #note: sub_data sorted by time
            sub_data = self.get_instances(self.data[:, i - moving_window: i, :]) #(t, x, y)
            (p_matrix, ind_matrix) = self.initialize_p(sub_data)
            old_p = np.zeros((len(p_matrix), len(p_matrix)), dtype = np.int16)
            l2_norm = np.sum((p_matrix - old_p) ** 2)
            list_l2_norm += [l2_norm]
            iter_num = 0
            while l2_norm > 0.01:
                print(i, iter_num)
                print(l2_norm)
                #step 1: background data and parent-child interpoint distances
                background = []
                offspring = []
                for col in range(0, len(p_matrix)):
                    ind = np.random.choice(list(range(0, len(p_matrix))), 
                                           p = p_matrix[:,col])
                    if ind == col:
                        background += [sub_data[ind,:]]
                    else:
                        offspring += [sub_data[col,:] - sub_data[ind,:]] #CHECK
                num_background += [len(background)]
                print(num_background)
                background = np.array(background)
                offspring = np.array(offspring)
                #step 2: fit kdes on background and offspring data
                (v, u, g) = self.train(background, offspring, fixed_bandwidth, bandwidth,
                                        u_k, v_k)
                #step 3: update P
                old_p = copy.deepcopy(p_matrix)
                for col in range(0, len(p_matrix)):
                    if ind_matrix[col] != -1:  #t_k < t events
                        ind = int(np.asscalar(ind_matrix[col]))
                        g_i = []
                        if fixed_bandwidth == False:
                            v_i = v(sub_data[col,0])
                            u_i = u(sub_data[col,1], sub_data[col, 2])
                            g_i = g(sub_data[col,:] - sub_data[0:ind,:])
                        else:
                            v_i = np.exp(v.score_samples(sub_data[col,0]))
                            u_i = np.exp(u.score_samples(sub_data[col,1:].reshape(1,-1)))
                            g_i = np.exp(g.score_samples(sub_data[col,:] - sub_data[0:ind,:]))    
                        if ind > 0:
                                p_matrix[0:ind, col] = g_i
                        p_matrix[ind + 1: len(p_matrix) - 1, col] = 0 #row > col vals
                        p_matrix[col, col] = u_i * v_i #row = col vals
                        p_matrix[:,col] = p_matrix[:,col]/np.sum(p_matrix[:,col])
                
                l2_norm = np.sum((p_matrix - old_p) ** 2)
                list_l2_norm += [l2_norm]
                iter_num += 1
            
            pred_sample = np.concatenate((self.data[:,i,1].reshape(len(self.data), 1),self.SDS.coords),axis = 1)
            if fixed_bandwidth == False:
                v_predict = v(pred_sample[:,0])
                u_predict = u(pred_sample[:,0], pred_sample[:,1])
                g_predict = g(pred_sample) #check if sum or not here
            else:
                v_predict = v.score_samples(pred_sample[:,0].reshape(-1,1))
                u_predict = u.score_samples(pred_sample[:,1:])
                g_predict = g.score_samples(pred_sample) #check if sum or not here
            predictions = u_predict * v_predict + g_predict
            #save results to frame
            results[row_num: row_num + n_space,0] = str(self.outfile)
            results[row_num: row_num + n_space,1] = counts[:,i,1].astype(str) 
            results[row_num: row_num + n_space,2] = counts[:,i,0].astype(str)
            results[row_num: row_num + n_space,3] = np.sum(counts[:,i, 2:], axis = 1).astype(str)
            results[row_num: row_num + n_space,4] = predictions.astype(str)
            row_num += 1  
        #writing results to file for output (for testing purposes) - this section will be removed in the future
        file.writerow("background") 
        file.writerows(background)
        file.writerow("\n")
        file.writerow("offspring")
        file.writerows(offspring)
        file.writerow("\n")
        print(list_l2_norm)
        file.writerow(list_l2_norm)
        file.writerow("\n")
        print(num_background)
        file.writerow(num_background)
        file.writerow("\n")
        print(np.std(background.astype(float), axis = 0))
        file.writerow(np.std(background.astype(float), axis = 0))
        file.writerow("\n")
        print(np.std(offspring.astype(float), axis = 0))
        file.writerow(np.std(offspring.astype(float), axis = 0))
        file.writerow("\n")
        file_obj.close()
        return results
                    
                    
                


=======
# -*- coding: utf-8 -*-
"""
Hawkes Process as per Mohler et al. (2011) - class file
Created on Wed Jul 11 11:20:59 2018

@author: inezk
"""
import numpy as np
from predictor_class import Predictor
from sklearn.neighbors.kde import KernelDensity
from scipy.spatial.distance import pdist, squareform
import copy
import csv

#note - (t, x,y) sorted by time

class hawkes_process(Predictor):
    def __init__(self, data, filename, moving_window = 4, fixed_bandwidth = True, bandwidth = 500, u_k = 15, 
                 v_k = 100, start_index = 104, end_index = 260, crime_types = []):
        Predictor.__init__(self, data, filename)
        self.model = ""
        self.predictions = self.predict(moving_window, start_index, end_index,
                                         fixed_bandwidth, bandwidth, u_k, v_k, crime_types)

    #getting v,u,g
    def train(self, background, offspring, fixed_bandwidth, bandwidth, u_k, v_k): #v, u , g
        if fixed_bandwidth == True:
            v = KernelDensity(bandwidth)
            u = KernelDensity(bandwidth)
            g = KernelDensity(bandwidth)
            
            v.fit(background[:, 0].reshape(-1, 1))
            u.fit(background[:, 1:])
            g.fit(offspring)
        else: #scale so variance is 1 for offspring and background data
            offspring_scaled = offspring/(np.var(offspring,axis = 1).reshape(len(offspring), 1))
            background_scaled = background/(np.var(background,axis = 1).reshape(len(background), 1))
            #get D, matrix of nearest neighbor distances to each data point
            offspring_scaled = squareform(pdist(offspring_scaled))
            background_scaled_v = squareform(pdist(background_scaled[:,0].reshape(len(background), 1)))
            background_scaled_u = squareform(pdist(background_scaled[:,1:]))
            offspring_scaled.sort(axis=1)
            background_scaled_v.sort(axis=1)
            background_scaled_u.sort(axis=1)
            #getting Di, the ith nearest neighbor distance
            Du_i = background_scaled_u[:, u_k]
            Dv_i = background_scaled_v[:, v_k]
            Dg_i = offspring_scaled[:, u_k].reshape(len(offspring),1)
            
            off_sig = np.std(offspring,axis=0, dtype=np.float64) #(t,x,y)
            back_sig = np.std(background, axis = 0, dtype = np.float64) #(t,x,y)
            
            def g(data):  #g(t,x,y)
                n_offspring = offspring.shape[0]
                n_cases = data.shape[0]
                n_dim = data.shape[1]
                #having all entries in matrix (n_cases, n_offspring, n_dim)
                const = (1/(np.prod(off_sig) * (2 * np.pi) ** (3/2) * Dg_i **3))
                const_tiled = np.tile(const, (n_cases, 1, 1)).reshape(n_cases,n_offspring)
                data_tiled = np.tile(data.reshape(n_cases, 1, n_dim), (1,n_offspring,1))
                exponent_denoms = np.tile((2 * (off_sig**2).reshape(1, n_dim) * Dg_i **2).reshape(1, n_offspring, n_dim), 
                                          (n_cases, 1, 1))
                exponents = (-((data_tiled - offspring)**2)/exponent_denoms).sum(axis=(2)).astype(float)
                final_g = (const_tiled*np.exp(exponents)).mean(axis=1)
                return final_g
            
            def v(t):
                const = 1/(back_sig[0] * (2 * np.pi) ** (1/2) * Dv_i)
                exp_t = -(t - background[:,0]) ** 2/(2 * back_sig[0] ** 2 * Dv_i)
                total = const * np.exp(exp_t.astype(float))
                final_v = np.sum(total)/(len(background))
                return final_v
            
            def u(x, y):
                const = 1/(back_sig[1] * back_sig[2] * (2 * np.pi) * Du_i ** 2)
                exp_x = -(x - background[:,1]) ** 2/(2 * back_sig[1] ** 2 * Du_i)
                exp_y = -(y - background[:,2]) ** 2/(2 * back_sig[2] ** 2 * Du_i)
                total = const * np.exp(exp_x.astype(float) + exp_y.astype(float))
                final_u = np.sum(total)/(len(background))
                return(final_u)
                
        return (v, u, g)
    
    def get_instances(self, frame): #converting from counts to instances 
        date_range = frame
        instances = np.sum(date_range[:,:,2:], axis = (2,1), dtype = np.int64)
        xtrain = np.repeat(self.SDS.coords, instances, axis = 0)
        time = np.repeat(date_range[:,:,1], instances, axis = 0)
        time_intsances = np.sum(date_range[:,:,2:], axis = 2, dtype = np.int64)
        time = time[:,np.where(time_intsances > 0)[1]][0]
        whole_frame = np.concatenate((time.reshape(len(time), 1), xtrain), axis = 1)
        whole_frame = whole_frame[whole_frame[:,0].argsort()] 
        return whole_frame
    
    #setting up p_matrix with random probabilities for events j (col)< i (row)
    def initialize_p(self, data):
        p = np.zeros((len(data), len(data)), dtype= float)
        ind_matrix = np.zeros((len(p), 1)) #matrix for storing indices t_k < t
        for col in range(0, len(p)):
            ind = np.where(data[:,0] < data[col,0])[0] 
            if len(ind) > 0: 
                ind = ind[len(ind) - 1] + 1
                ind_matrix[col] = ind
                p[0:ind, col] = np.random.random(ind)
                p[col, col] = 0
                p[:, col] = p[:, col]/(np.sum(p[:,col])* 2)
                p[col, col] = 0.5
            else: 
                p[col, col] = 1
                ind_matrix[col] = -1 #for when no events appear before it
        return (p, ind_matrix)
    
    def predict(self, moving_window, start_index, end_index, fixed_bandwidth, bandwidth,
                u_k, v_k, crime_types):
        if len(crime_types) > 0:
            self.data = self.data[:,:, [0, 1] + crime_types]
        n_space = len(self.data)
        results = np.zeros(((end_index - start_index) * n_space, 5), dtype = object)
        #these lines are for testing purposes - will eventually be removed
        if fixed_bandwidth: file_obj = open("sample_results_true.csv", "w") 
        else: file_obj = open("sample_results_false.csv", "w")
        file = csv.writer(file_obj)
        row_num = 0
        #initializing fields
        counts = self.data
        list_l2_norm = []
        num_background = []
        for i in range(start_index, end_index): 
            #note: sub_data sorted by time
            sub_data = self.get_instances(self.data[:, i - moving_window: i, :]) #(t, x, y)
            (p_matrix, ind_matrix) = self.initialize_p(sub_data)
            old_p = np.ones((len(p_matrix), len(p_matrix)), dtype = np.int16)
            l2_norm = np.sum((p_matrix - old_p) ** 2)
            list_l2_norm += [l2_norm]
            iter_num = 0
            while l2_norm > 0.001:
                print(i, iter_num)
                #step 1: background data and parent-child interpoint distances
                background = []
                offspring = []
                for col in range(0, len(p_matrix)):
                    ind = np.random.choice(list(range(0, len(p_matrix))), 
                                           p = p_matrix[:,col])
                    if ind == col:
                        background += [sub_data[ind,:]]
                    else:
                        offspring += [sub_data[col,:] - sub_data[ind,:]] #CHECK
                num_background += [len(background)]
                print(num_background)
                background = np.array(background)
                offspring = np.array(offspring)
                #step 2: fit kdes on background and offspring data
                (v, u, g) = self.train(background, offspring, fixed_bandwidth, bandwidth,
                                        u_k, v_k)
                #step 3: update P
                old_p = copy.deepcopy(p_matrix)
                for col in range(0, len(p_matrix)):
                    if ind_matrix[col] != -1:  #t_k < t events
                        ind = int(np.asscalar(ind_matrix[col]))
                        g_i = []
                        if fixed_bandwidth == False:
                            v_i = v(sub_data[col,0])
                            u_i = u(sub_data[col,0], sub_data[col, 1])
                            g_i = g(sub_data[col,:] - sub_data[0:ind,:])
                        else:
                            v_i = np.exp(v.score_samples(sub_data[col,0]))
                            u_i = np.exp(u.score_samples(sub_data[col,1:].reshape(1,-1)))
                            g_i = np.exp(g.score_samples(sub_data[col,:] - sub_data[0:ind,:]))    
                        if ind > 0:
                                p_matrix[0:ind, col] = g_i
                        p_matrix[ind + 1: len(p_matrix) - 1, col] = 0 #row > col vals
                        p_matrix[col, col] = u_i * v_i #row = col vals
                        p_matrix[:,col] = p_matrix[:,col]/np.sum(p_matrix[:,col])
                
                l2_norm = np.sum((p_matrix - old_p) ** 2)
                list_l2_norm += [l2_norm]
                iter_num += 1
            
            pred_sample = np.concatenate((self.data[:,i,1].reshape(len(self.data), 1),self.SDS.coords),axis = 1)
            if fixed_bandwidth == False:
                v_predict = v(pred_sample[:,0])
                u_predict = u(pred_sample[:,0], pred_sample[:,1])
                g_predict = g(pred_sample) #check if sum or not here
            else:
                v_predict = v.score_samples(pred_sample[:,0].reshape(-1,1))
                u_predict = u.score_samples(pred_sample[:,1:])
                g_predict = g.score_samples(pred_sample) #check if sum or not here
            predictions = u_predict * v_predict + g_predict
            #save results to frame
            results[row_num: row_num + n_space,0] = str(self.outfile)
            results[row_num: row_num + n_space,1] = counts[:,i,1].astype(str) 
            results[row_num: row_num + n_space,2] = counts[:,i,0].astype(str)
            results[row_num: row_num + n_space,3] = np.sum(counts[:,i, 2:], axis = 1).astype(str)
            results[row_num: row_num + n_space,4] = predictions.astype(str)
            row_num += 1  
        #writing results to file for output (for testing purposes) - this section will be removed in the future
        file.writerow("background") 
        file.writerows(background)
        file.writerow("\n")
        file.writerow("offspring")
        file.writerows(offspring)
        file.writerow("\n")
        print(list_l2_norm)
        file.writerow(list_l2_norm)
        file.writerow("\n")
        print(num_background)
        file.writerow(num_background)
        file.writerow("\n")
        print(np.std(background.astype(float), axis = 0))
        file.writerow(np.std(background.astype(float), axis = 0))
        file.writerow("\n")
        print(np.std(offspring.astype(float), axis = 0))
        file.writerow(np.std(offspring.astype(float), axis = 0))
        file.writerow("\n")
        file_obj.close()
        return results
                    
                    
                


>>>>>>> a83aa26439585374f000de4f40575c539f600084
