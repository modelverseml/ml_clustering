# Ml Clustering

In machine learning, we usually deal with two main types of problems — supervised and unsupervised learning.

In supervised learning, we work with labeled data, which means we already know the correct outputs. This makes it easier to train and fine-tune models since we can directly measure how well they perform. Examples include predicting house prices or classifying whether an email is spam or not.

On the other hand, unsupervised learning is used when there are no labels or known outputs. Here, the goal is to explore the data and find hidden patterns or relationships within it.

A common example of unsupervised learning is clustering, where we group similar data points together based on their characteristics. For instance, recommendation systems like YouTube ads or music suggestions on Spotify use clustering to group users with similar interests or behavior.

In short, clustering helps us make sense of unlabeled data by organizing it into meaningful groups — which can then be used for insights, recommendations, or decision-making.

 
<p align="center">
<img src="Images/clustering.png" alt="clustering" width="50%"/>
</p>

There are numerous algorithms available to solve clustering problems, and each uses a different approach to group data points. Below are some of the most commonly used clustering algorithms categorized by their underlying method.


## Clustering Algorithms

- Centroid-Based Methods

  These algorithms group data points based on their distance from central points (called centroids).
    - K-Means
    - Mini-Batch K-Means
    - K-Medoids
 
- Density-Based Methods

  These identify clusters as regions of high data density separated by regions of low density.
    - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    - OPTICS (Ordering Points To Identify the Clustering Structure)
    - HDBSCAN (Hierarchical DBSCAN)

- Hierarchical Methods

  These algorithms build a tree-like structure of clusters, showing how clusters are formed or merged at different levels.
    - Agglomerative Clustering (bottom-up approach)
    - Divisive Clustering (top-down approach)
    
- Distribution-Based Methods

  These assume that data points are generated from specific probability distributions, such as Gaussian distributions.
    - Gaussian Mixture Models (GMMs)

- Other Popular Methods
    - Mean Shift – shifts data points toward the nearest cluster center.
    - Affinity Propagation – identifies exemplars among data points and forms clusters around them.
    - BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) – efficient for large datasets.
    - Spectral Clustering – uses graph theory and eigenvalues to cluster data in complex structures.
    
