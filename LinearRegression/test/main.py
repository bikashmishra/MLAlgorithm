from FileRead import readASCIIintoArray
from LinearRegression import linearregression as lr
import numpy as np
import FeatureScaling as fs

def main():
    print 'TEST1'
    filenm = "data.txt"
    outdata = readASCIIintoArray(filenm, 0)
    x = np.asmatrix(outdata[:,1]).T
    y = np.asmatrix(outdata[:,2]).T
    x, xmean, xsig = fs.MeanNormalizeMatrix(x)
    y, ymean, esig = fs.MeanNormalizeArray(y)
    my_lr1 = lr()
    my_lr1.train(x,y, method = 'cg', maxiter=500,alpha=1.0)
    print my_lr1.gettheta()
    
    print 'TEST2'
    filenm = "datamulti.txt"
    outdata = readASCIIintoArray(filenm, 0)
    x = np.asmatrix(outdata[:,1:3])
    y = np.asmatrix(outdata[:,3]).T
    x, xmean, xsig = fs.MeanNormalizeMatrix(x)
    y, ymean, esig = fs.MeanNormalizeArray(y)
    my_lr2 = lr()
    my_lr2.train(x,y,method = 'cg',maxiter=1000,alpha=1.0)
    print my_lr2.gettheta()

#     print 'TEST 3'
#     x = np.asmatrix( [[0.9, 0], [0.1, 1.0]] ) 
#     y = np.asmatrix([5.0, 0.0]).T
#     my_lr3 = lr()
#     my_lr3.train(x,y)
#     print my_lr3.gettheta()
    
if __name__ == "__main__":
    main()
    

