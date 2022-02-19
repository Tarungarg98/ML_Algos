import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time

## Buiding KMeans algorithms from scrath
class KMeans_TG:
    ## coordinates : Data points on which clusters are to be made
    ## no_of_clusters : Total number of clusters to be formed, default 3
    ## plot : 0 - no plot, 1 - original data points with centroids, 2 - plot clusters (3D can be done too)
    ## max_iter : max number iterations to try for stable clusters
    def __init__(self,coordinates,no_of_clusters=3,plot=0,max_iter=30):
        self.max_iter=max_iter
        self.plot=plot

        # Convert coordinates to numpy array
        if type(coordinates)!=np.ndarray:
            coordinates=np.array(coordinates)
            
        # Dimension of dataset    
        n_d=coordinates.shape[1]
        centroids=[]
        self.n_d=n_d
        
        # Initializing centroids for all dimension
        for d in range(n_d):
            min_val=min(coordinates[:,d])
            max_val=max(coordinates[:,d])
            
            # Generating random centroids within the dataset range
            centroids_d=KMeans_TG.generate_centroids(min_val,max_val,no_of_clusters)
            centroids.append(centroids_d)
            
        # Transposing it to align at coordinates level
        centroids=np.array(centroids).T
        self.coordinates=coordinates
        self.centroids=centroids
        self.no_of_clusters=no_of_clusters

    # Generating random centroids    
    def generate_centroids(min_val,max_val,no_of_points):
        coordinates=[]
        for n in range(no_of_points):
            d=random.randint(min_val,max_val)
            coordinates.append(d)
        return coordinates

    # Calculating sum of distances between new and old centroids
    def cal_diff(self):
        dist=0
        for centroid in range(len(self.centroids)):
            dist+=math.dist(self.new_centroids[centroid],self.centroids[centroid])
        return dist
    
    def KMeans_loop(self,count):
        if self.plot==1:
            plt.clf()
            colours=['red','yellow','blue','black','orange']
            plt.scatter(self.coordinates[:,0],self.coordinates[:,1],color='green')
            for x in range(self.no_of_clusters):
                plt.scatter(self.centroids[x][0],self.centroids[x][1],color=colours[x%5])
            plt.show()
        cluster_dict={}
        centroids_data={}
        for centroid in range(len(self.centroids)):
            cluster_dict[centroid]=[]
            centroids_data[centroid]=self.centroids[centroid]
        for pair in self.coordinates:
            min_dist=100000
            cluster=0
            for centroid in range(len(self.centroids)):
                dist=math.dist(pair,self.centroids[centroid])
                if dist<min_dist:
                    min_dist=dist
                    cluster=centroid
            cluster_dict[cluster].append(pair)

        new_centroids=[]
        self.cluster_dict=cluster_dict
        for centroid in range(len(self.centroids)):
            cluster_dict[centroid]=np.array(cluster_dict[centroid])
            new_centroids_d=[]
            for d in range(self.n_d):
                mean_val=cluster_dict[centroid][:,d].mean()
                new_centroids_d.append(mean_val)        
            new_centroids.append(tuple(new_centroids_d))
        self.new_centroids=new_centroids
        if self.plot==2:
            if self.n_d>=3:
                print("Cannot plot more than 3 dimensions")
            plt.clf()
            colours=['red','yellow','blue','black','orange','cyan','magenta','white','green']
            if self.n_d==3:
            
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.set_zlabel('z-axis')
            
            for x in range(self.no_of_clusters):
                if self.n_d==3:
                    ax.scatter(cluster_dict[x][:,0],cluster_dict[x][:,1],cluster_dict[x][:,2],colours[x])
                if self.n_d==2:
                    plt.scatter(cluster_dict[x][:,0],cluster_dict[x][:,1],color=colours[x])
                if self.n_d==1:
                    plt.scatter(cluster_dict[x].flatten(),[0]*len(cluster_dict[x]),color=colours[x])
            plt.show()
        self.cluster_dict=cluster_dict

    def main(self):
        start = time.time()

        print("Fitting ",self.no_of_clusters," clusters")
        print("Shape of input data : ",self.coordinates.shape)
        count=1
        print("Running iteration :",count)
        self.KMeans_loop(count)
        print("Total Distance between coordinates :",self.cal_diff())
        while not np.array_equal(self.new_centroids,self.centroids):
            self.centroids=self.new_centroids
            count+=1
            print("Running iteration :",count)
            self.KMeans_loop(count)
            print("Total Distance between coordinates :",self.cal_diff())
            if count>=self.max_iter:
                break
        end = time.time()
        print("Total time taken (in seconds) : ",round(end - start,2))
        return
            
            

            
    
    