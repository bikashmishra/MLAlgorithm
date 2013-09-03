import numpy as np
import matplotlib.pyplot as plt

class kmeanscluster():
    """ k-means clustering algorithm for features in R^n"""
    
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    
    def kmeans(self, X, nclusters, maxiter=100, eps=1.0e-06):
        """ X is array_like of size [npoints] x [nfeatures]"""
        print 'Clustering...'       
            
        self.randInitCentroids(X, nclusters)
        self.findClosestCentroid(X)  
        self.findClusterMean(X)
        initcost = self.cost(X)
        itr = 1
        relcost=1.0
        cost = 0.0
        print 'iter %d, Cost %f' % (itr,relcost)
        while itr < maxiter and relcost > eps:
            self.findClosestCentroid(X)
            self.findClusterMean(X)
            newcost = self.cost(X)
            relcost = np.absolute(newcost-cost)/initcost
            cost = newcost
            print 'iter %d, Cost %f, Rel.Cost %f' % (itr, cost,relcost)
            itr += 1
        print 'Done'
       
    def getCentroids(self):
        return self.centroids
    
    def getClusterAssignment(self):
        return self.centroid_list
         
    def randInitCentroids(self, X, n=1):
        self.centroids = np.random.permutation(X)[0:n,:]
       
    def distance(self, x1, x2):
        return self.euclidean_distance(x1, x2)
    
    def  euclidean_distance(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("Cannot calculate distance between vectors of dimension %d and %d" % (len(x1), len(x2)))
        return np.linalg.norm(x1-x2)
    
    def findClosestCentroid(self, X):
        npts = X.shape[0]
        ncentroids = self.centroids.shape[0]
        self.centroid_list = np.zeros(npts, dtype=np.int64)
        for i in range(npts):
            mindist=10000.0
            for j in range(ncentroids):
                dist = self.distance(X[i], self.centroids[j])
                if dist < mindist :
                    mindist = dist
                    self.centroid_list[i] = j
        
    def findClusterMean(self, X):
        nclusters = self.centroids.shape[0]
        for i in range(nclusters):
            self.centroids[i] = np.mean(X[self.centroid_list == i,:], axis=0)

    def cost(self, X):
        carr = [self.centroids[i] for i in self.centroid_list]
        cost = np.linalg.norm(X-carr)
        return cost

    def plot2d(self,X):
        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)     
        ax1.scatter(X[:,0], X[:,1], c=self.colors[self.centroid_list], s=30)
        ax1.scatter(self.centroids[:,0], self.centroids[:,1], c=self.colors[range(len(self.centroids))], s=300, marker='x')
        plt.show()