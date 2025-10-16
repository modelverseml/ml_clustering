
import numpy as np

class Kmeans:

    """
        Inintialising all the variables
    """
    def __init__(self, n_clusters,data):
        
        self.n_clusters = n_clusters
        self.data = data.copy()
        self.columns = data.columns.tolist()
        self.data['cluster'] = np.zeros(data.shape[0])
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

    def assign_clusters(self):

        for i in range(100):

            self.data['cluster'] = self.data[self.columns].apply(self.get_single_instance_cluster,axis=1)

            self.update_centroids()

            updated_cost_value = self.get_cost_function_value()

            if self.error > (self.cost_function_val -  updated_cost_value): 
                print(i)
                break
            
            self.cost_function_val = updated_cost_value


        return self.centers,self.data['cluster']

    """
    Update single instance cluster
    """
    def get_single_instance_cluster(self,instance):
        
        return np.argmin([np.sum((instance.values - center) ** 2) for center in self.centers])

    """
    Updating centroids
    """
    def update_centroids(self):

        self.centers = self.data.groupby('cluster')[self.columns].mean().values
    
    """
    Getting cost function
    """
    def get_cost_function_value(self):
        cost = 0
        for index,center in enumerate(self.centers):
            for _, instance in (self.data[self.data['cluster']==index][self.columns]).iterrows():
                cost += sum((instance - center)**2)

        return round(cost,2)