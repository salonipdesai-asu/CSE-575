#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:32:49 2019

@author: salonidesai
"""

#KMEANS ALGORITHM - Strategy 2
import scipy.io
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt 
Numpyfile= scipy.io.loadmat('/Users/salonidesai/Downloads/AllSamples.mat') 
Samples_X = []
Samples_Y = []
Samples = Numpyfile.get('AllSamples')

for sample in Samples:
    Samples_X.append(sample[0])
    Samples_Y.append(sample[1])
Objective_Function_Values1=[]
Objective_Function_Values2=[]
for x in range(0,2):
    
    K_Values=[2,3,4,5,6,7,8,9,10]
    #implementing for k= n
    for k in range(2,11):
        print("K value:",k)
        mu =[]
        #Selecting initial samples from clusters
        indices =np.random.choice(300,size=1,replace=False, p=None)
    
        for index in indices:
            mu.append((Samples[index][0], Samples[index][1]))

        
        for i in range(1,k):
            strategy2 =[]
            for sample in Samples:
                distan = 0
                for center in mu:
                    distan = distan + distance.euclidean(sample,center)
                    
                avg_distance = distan/len(mu)
                strategy2.append(avg_distance)
            
            max_dist_index = strategy2.index(max(strategy2))
            #print(Samples[max_dist_index])
            mu.append(Samples[max_dist_index])
    
            
        print("The Initial Cluster Centers chosen are:", mu)
        print("")
        
        final_cluster_centers =[]
        final_cluster_matrix =[]
        Stopping_Condition = False
        while(Stopping_Condition!=True):
            #create a distances matrix
            distances = np.ndarray(shape=(300,k),dtype=float)
            #print(np.shape(distances))
    
            #For each datapoint, calculate the distance from each cluster centre
    
            for i in range(len(Samples)):
                for j in range(k):
                    dist= distance.euclidean(Samples[i],mu[j])
                    distances[i][j] = dist
            #print(distances)        
            #Assign each datapoint to the nearest cluster center
    
            cluster_index=[]
    
            for i in range(len(distances)):
                distance_list = distances[i].tolist()
                min_index = distance_list.index(min(distance_list))
                cluster_index.append(min_index)
            
            #print(len(cluster_index))
        
    
            #Recalculate centroids/ cluster centroids
            new_mu =[]
        
            cluster_matrix=[]
            for i in range(k):
                cluster=[]
                new_centre =(0,0)
                sumx= 0
                sumy= 0
        
                for j in range(len(cluster_index)):
                    if (i == cluster_index[j]):
                        cluster.append(Samples[j])
                    
            
                for l in range(len(cluster)):
                    sumx = sumx + cluster[l][0]
                    sumy = sumy + cluster[l][1]
                if len(cluster)!=0:
                    new_centre = (sumx/len(cluster), sumy/len(cluster))
                    new_mu.append(new_centre)
                    cluster_matrix.append(cluster)
                else:
                    new_mu.append(mu[i])
            
            #print(new_mu)
    
            #if old cluster centers are same as new cluster centers, stop, otherwise continue
    
        
            count =0
    
            for i in range(k):
                if (new_mu[i][0] == mu[i][0] and new_mu[i][1] == mu[i][1]):
                    count = count + 1
    
            if (count == k):
                Stopping_Condition = True
                final_cluster_centers= new_mu
                final_cluster_matrix = cluster_matrix
    
            #if stopping condition has not been reached, assign new clusters centers to  mu 
            if Stopping_Condition == False:
                for i in range(k):
                    mu[i] = new_mu[i]
    
        #Printing the results
        print("The Final cluster centers are:",final_cluster_centers)
        print("")
        for cl in range(k):
            print("Cluster",(cl+1))
            print("")
            print(final_cluster_matrix[cl])
            print("")
    

        
        #Calculating the Objective Function value
        obj_value =0.0
        for i in range(k):
            current_center = final_cluster_centers[i]
            for sample in final_cluster_matrix[i]:
                obj_value = obj_value + (distance.euclidean(sample, current_center) * distance.euclidean(sample, current_center))
        if(x==0):
            Objective_Function_Values1.append(obj_value)
        else:
            Objective_Function_Values2.append(obj_value)

    
#Plotting Objective Function vs k values
print("For 1st Random Initialization")
plt.plot(K_Values,Objective_Function_Values1)
plt.xlabel('No of Clusters') 
plt.ylabel('Objective Function') 
plt.show()

print("For 2st Random Initialization")
plt.plot(K_Values,Objective_Function_Values2)
plt.xlabel('No of Clusters') 
plt.ylabel('Objective Function') 
plt.show()
