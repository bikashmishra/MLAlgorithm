# Takes in x-independent params matrix, y-target variable array
# and returns theta-array

# x(nexamples, nfeatures)
# y(nexamples)

import numpy as np
from scipy import optimize

def calcHypothesis(x, theta):
    return x*np.asmatrix(theta).T
    
def costFunction(theta, *args):
    x,y = args
    m = x.shape[0]
    h = calcHypothesis(x, theta)
    J = (1./(2.*m)*(h-y).T*(h-y))[0][0]
    return J

def costFunctionGrad(theta, *args):
    x,y = args
    m = x.shape[0]
    h = calcHypothesis(x, theta)
    grad = 1./m*x.T*(h-y)
    return np.asarray(grad)[:,0] # return value should be a 1D array of size (ntheta,)
    
def LinearRegGradientDescent(x, y, alpha, niters):
    ntheta = x.shape[1]
    theta = np.zeros(ntheta)
    J_hist = np.zeros(niters)
    args = (x,y)
    for i in range(niters):
        J=np.asmatrix(costFunction(theta, *args))
        grad=np.asmatrix(costFunctionGrad(theta, *args))
        theta = theta - alpha*grad
        J_hist[i]= J
    return (theta, J_hist) 
 
# x, y are of type matrix. theta is of type array
def LinearReg(x,y, method='cg', alpha=1.0, maxiter=None, tol=1e-03):
    col_one = np.asmatrix(np.ones(x.shape[0])).T
    x = np.c_[col_one, x]
    ntheta = x.shape[1]
    theta0 = np.zeros(ntheta)
    # choose the method
    if method == 'cg':
        theta = optimize.fmin_cg(costFunction, theta0, fprime=costFunctionGrad, args=(x,y), gtol=tol, maxiter=maxiter)
        return theta
    elif method == 'bfgs':
        theta = optimize.fmin_bfgs(costFunction, theta0, fprime=costFunctionGrad, args=(x,y), gtol=tol, maxiter=maxiter)
        return theta
    elif method == 'gd':
        theta, J_hist = LinearRegGradientDescent(x, y, alpha, maxiter)
        return theta
    