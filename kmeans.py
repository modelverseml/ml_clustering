
import numpy as np
import pandas as pd

class ManualKMeans:

    """
        Inintialising all the variables
    """
    def __init__(self,data):
        
        self.data = data.copy()
        self.cost_function_val = np.inf

        self.error = 0.1



    """
    Intialising the centers we are getting this from user input becase we can use the same functionality for 
    both k-means and k++

    """
    def initialise_centers(self, centers = []):

        self.centers = centers

    """
    Here we are assigning clusters to the datapoints based on the nearest cluster center
    """

    def assign_clusters(self,max_iterations = 50,batch_size = None):

        if not batch_size:

            batch_size = self.data.shape[0]

        for i in range(max_iterations):

            batch_indices = np.random.choice(len(self.data), batch_size, replace=False)

            batch = self.data.iloc[batch_indices]
            
            cols = batch.columns

            clusters = batch.apply(self.get_single_instance_cluster, axis=1)
            
            self.update_centroids(batch,clusters)

            updated_cost_value = self.get_cost_function_value(batch,clusters)


        final_clusters = self.data.apply(self.get_single_instance_cluster, axis=1)

        return updated_cost_value,self.centers,final_clusters



    """
    Update single instance cluster
    """
    def get_single_instance_cluster(self,instance):
        
        return np.argmin([np.sum((instance.values - center) ** 2) for center in self.centers])

    """
    Updating centroids
    """
    def update_centroids(self,batch,clusters):

        updated_data = batch.copy()

        updated_data['cluster'] = clusters

        self.centers = updated_data.groupby('cluster').mean().values
    
    """
    Getting cost function
    """
    def get_cost_function_value(self,batch,clusters):
        cost = 0

        for index,center in enumerate(self.centers):
            for _, instance in (batch[clusters==index]).iterrows():
                cost += sum((instance - center)**2)

        return round(cost,2)