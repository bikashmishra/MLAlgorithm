from FileRead import readASCIIintoArray
import LinearRegression as lr
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
    print lr.LinearReg(x,y, method = 'cg', maxiter=500,alpha=1.0)
    
    print 'TEST2'
    filenm = "datamulti.txt"
    outdata = readASCIIintoArray(filenm, 0)
    x = np.asmatrix(outdata[:,1:3])
    y = np.asmatrix(outdata[:,3]).T
    x, xmean, xsig = fs.MeanNormalizeMatrix(x)
    y, ymean, esig = fs.MeanNormalizeArray(y)
    print lr.LinearReg(x,y,method = 'cg',maxiter=1000,alpha=1.0)
    
if __name__ == "__main__":
    main()
    

