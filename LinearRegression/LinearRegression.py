# Takes in x-independent params matrix, y-target variable array
# and returns theta-array

# x(nexamples, nfeatures)
# y(nexamples)

import numpy as np
from scipy import optimize

class linearregression():
    
    def calcHypothesis(self, x, theta):
        return x*np.asmatrix(theta).T
    
    def costFunction(self, theta, *args):
        x,y = args
        m = x.shape[0]
        h = self.calcHypothesis(x, theta)
        J = (1./(2.*m)*(h-y).T*(h-y))[0][0]
        return J

    def costFunctionGrad(self, theta, *args):
        x,y = args
        m = x.shape[0]
        h = self.calcHypothesis(x, theta)
        grad = 1./m*x.T*(h-y)
        return np.asarray(grad)[:,0] # return value should be a 1D array of size (ntheta,)
    
    def LinearRegGradientDescent(self, x, y, alpha, niters):
        ntheta = x.shape[1]
        theta = np.zeros(ntheta)
        J_hist = np.zeros(niters)
        args = (x,y)
        for i in range(niters):
            J=np.asmatrix(self.costFunction(theta, *args))
            grad=np.asmatrix(self.costFunctionGrad(theta, *args))
            theta = theta - alpha*grad
            J_hist[i]= J
        return (theta, J_hist) 
 
    def gettheta(self):
        return self.theta
    
    # x, y are of type matrix. theta is of type array
    def train(self, x,y, method='cg', alpha=1.0, maxiter=None, tol=1e-03):
        col_one = np.asmatrix(np.ones(x.shape[0])).T
        x = np.c_[col_one, x]
        ntheta = x.shape[1]
        theta0 = np.zeros(ntheta)
        # choose the method
        if method == 'cg':
            self.theta = optimize.fmin_cg(self.costFunction, theta0, fprime=self.costFunctionGrad, args=(x,y), gtol=tol, maxiter=maxiter)
        elif method == 'bfgs':
            self.theta = optimize.fmin_bfgs(self.costFunction, theta0, fprime=self.costFunctionGrad, args=(x,y), gtol=tol, maxiter=maxiter)
        elif method == 'gd':
            self.theta, J_hist = self.LinearRegGradientDescent(x, y, alpha, maxiter)
    
    
    def predict(self,x):
        x = np.c_[1, x]
        if len(x) != len(self.theta):
            raise ValueError("length of theta and x are %d, %d" %(len(self.theta), len(x)))
        return self.theta*x.T