import numpy as np
import kmeans

def kmeanstest():
    nclusters = 5
    x = np.random.random_sample((2000,2))*10
    my_kmc = kmeans.kmeanscluster()
    my_kmc.kmeans(x, nclusters)
    my_kmc.plot2d(x)

if __name__ == "__main__":
    kmeanstest()