# This reads files of different formats for analysis
# Retun values are typically numpy objects


import numpy as np

# read csv file for analysis and output an array
def readCSVintoArray(filename, skipheader):
    with open(filename,"r") as csvfile:
        data = np.loadtxt(csvfile, delimiter=',', skiprows=skipheader)
    return data

# read an ASCII file and output an array. Default delimiter is one space
def readASCIIintoArray(filename, skipheader, delim=' '): 
    with open(filename,"r") as inputfile:
        data = np.loadtxt(inputfile, delimiter=' ', skiprows=skipheader)
    return data