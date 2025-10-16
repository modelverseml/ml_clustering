
import numpy as np

class Kmeans:

    def __init__(self, n_clusters,data):

        self.n_clusters = n_clusters
        self.data = data
        self.clusters = np.zeroes(data.shape[0])

    def intialise_centers(self, centers = []):

        self.centers = centers

    def get_cetroids