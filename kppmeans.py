from kmeans import ManualKMeans
import numpy as np


class KPPMeans:

    def __init__(self,n_clusters, data):

        self.n_clusters = n_clusters
        self.data = data
        self.get_initial_centers()

    
    def get_initial_centers(self):
        
        n_samples = self.data.shape[0]
        first_idx = np.random.choice(n_samples)
        
        self.centers = [self.data.loc[first_idx].values.tolist()]

        for _ in range(1, self.n_clusters):

            ecludian_dist = np.array([min(sum((center-instance)**2) for center in self.centers) 
                                      for indx,instance in self.data.iterrows()])

            instance_probabilites = ecludian_dist/sum(ecludian_dist)

            self.centers.append(self.data.loc[np.random.choice(n_samples,p=instance_probabilites)].values.tolist())

    def assign_clusters(self):

        manual_kmeans_algo = ManualKMeans(data = self.data)
        manual_kmeans_algo.initialise_centers(self.centers)   
        return manual_kmeans_algo.assign_clusters()

