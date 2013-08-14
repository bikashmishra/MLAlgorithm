from FileRead import readCSVintoArray
import LogisticReg as lr
import numpy as np

def logregtest():
    print 'TESTING LOGISTIC REGRESSION'
    filenm = "logreg_data.txt"
    outdata = readCSVintoArray(filenm, 0)
    x = np.asmatrix(outdata[:,0:2])
    y = np.asmatrix(outdata[:,2]).T

    print lr.LogisticReg(x,y,method = 'downhill',maximiter=400)