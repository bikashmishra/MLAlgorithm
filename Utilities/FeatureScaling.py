import numpy as np

# Input is numpy array type
def MeanNormalizeArray(arrdata):
    dmean = np.mean(arrdata)
    #dmax  = np.max(arrdata)
    #dmin  = np.min(arrdata)
    sig = np.std(arrdata)
    nordata = (arrdata - dmean)/sig
    return (nordata, dmean, sig)

#Input is 2D array 
# Features are along cols and number of examples=number of rows 
def MeanNormalizeMatrix(arrdata):
    dmean = np.mean(arrdata, axis=0)
    #dmax  = np.amax(arrdata, axis=0)
    #dmin  = np.amin(arrdata, axis=0)
    sig = np.std(arrdata, axis=0)
    nordata = (arrdata-dmean)/sig
    return (nordata, dmean, sig)

"""
def main():
    A = np.array([[3,5],[2,7]])
    print A
    print MeanNormalizeMatrix(A)

if __name__ == "__main__":
    main()
"""
