import numpy as np
import pandas as pd

class DBSCAN:

    def __init__(self,data,epsilon,MinPts):

        self.data = data
        self.epsilon = epsilon
        self.MinPts = MinPts

        # Use only numeric features for distance calculation
        self.cols_to_use = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Initialize helper columns
        self.data['is_visited'] = 0
        self.data['cluster'] = -1


    def get_clusters(self): 
        
        cluster_id = -1
        
        for index in self.data.index:

            if self.data.loc[index,'is_visited'] == 0:

                self.data.loc[index,'is_visited'] = 1
                
                neighbour_indices = self.get_neighbours_index(index)
                
                # Checking for core point
                if len(neighbour_indices)>= self.MinPts:
                    cluster_id+=1

                    self.data.at[index,'cluter'] = cluster_id
                    self.expand_cluster(neighbour_indices,cluster_id)

                # making it as noise for time being
                else:
                    self.data.loc[index,'cluster'] = -1
                
        return self.data['cluster']


    def expand_cluster(self, neighbour_indices,cluster_id):

        ## looping through neighbours and adding its neighbours to the list if it is core point
        while(neighbour_indices):

            neighbour_index = neighbour_indices.pop(0)

            if self.data.at[neighbour_index,'is_visited'] == 0:
                self.data.at[neighbour_index,'is_visited'] = 1

                neighbour_neighbours_indices = self.get_neighbours_index(neighbour_index)

                if len(neighbour_neighbours_indices) >= self.MinPts:
                    self.data.at[neighbour_index,'cluter'] = cluster_id

                    neighbour_indices.extend(neighbour_neighbours_indices)
            
            if self.data.at[neighbour_index,'cluster'] == -1:
                self.data.at[neighbour_index,'cluster'] = cluster_id


    def get_neighbours_index(self,index):
        
        instace  = self.data.loc[index,self.cols_to_use].values

        distances = np.sqrt(((self.data[self.cols_to_use].values -instace)**2).sum(axis=1))

        return self.data[distances<=self.epsilon].index.tolist()

        


