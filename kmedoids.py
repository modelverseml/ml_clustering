import numpy as np

class KMedoids:

    def __init__(self,n_clusters,data):

        self.data = data
        self.n_clusters = n_clusters
        self.get_initial_centers()

    def get_initial_centers(self):
        
        n_samples = self.data.shape[0]
        first_idx = np.random.choice(n_samples)
        
        self.medoids = [self.data.iloc[first_idx].values.tolist()]

        for _ in range(1, self.n_clusters):

            ecludian_dist = np.array([
                min(np.sum((mediod-instance)**2) for mediod in self.medoids) 
                for indx,instance in self.data.iterrows()
                ])

            instance_probabilites = ecludian_dist/ecludian_dist.sum()

            self.medoids.append(self.data.iloc[np.random.choice(n_samples,p=instance_probabilites)].values.tolist())


    def assign_medoids(self,max_iterations = 50,batch_size = None):

        if not batch_size:

            batch_size = self.data.shape[0]


        for i in range(max_iterations):

            batch_indices = np.random.choice(len(self.data), batch_size, replace=False)

            batch = self.data.iloc[batch_indices].reset_index(drop=True)
            
            batch_values = batch.values.astype(float)
            clusters = np.array([self.get_single_instance_cluster(row) for row in batch_values])
            
            self.update_medoids(batch,clusters)

            updated_cost_value = self.get_cost_function_value(batch,clusters)

        final_clusters = np.array([self.get_single_instance_cluster(row) for row in self.data.values])

        return updated_cost_value,self.medoids,final_clusters
    
    

    
    """
    Update single instance cluster
    """
    def get_single_instance_cluster(self,instance):
        
        dists = np.sum((self.medoids - instance) ** 2, axis=1)

        return int(np.argmin(dists))
    
    
    
    def update_medoids(self,batch,clusters):

        final_medoids = []
        for cluster_idx in range(self.n_clusters):
            mask = (clusters == cluster_idx)
            cluster_points = batch.iloc[np.where(mask)[0]]

            medoids_distance = np.inf
            medoids = []
            
            for instance in cluster_points.values:
                instance_dis = np.sum(np.sum((cluster_points-instance)**2,axis =1))

                if instance_dis < medoids_distance :
                    medoids_distance = instance_dis
                    medoids = instance

            final_medoids.append(medoids)
        self.medoids = np.array(final_medoids)


    """
    Getting cost function
    """
    def get_cost_function_value(self,batch,clusters):
        cost = 0
        for index,mediod in enumerate(self.medoids):
            mask = (clusters == index)
            pts = batch.iloc[np.where(mask)[0]].values

            cost += np.sum(np.sum((pts - mediod) ** 2, axis=1))
        
        return round(cost,2)