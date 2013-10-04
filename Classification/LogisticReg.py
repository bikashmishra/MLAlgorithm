import numpy as np
from scipy import optimize

class logisticregression():
    def DecisionBoundary(self, x, theta):
        return x*np.asmatrix(theta).T

    def calcHypothesis(self, x, theta):
        z = self.DecisionBoundary(x, theta)
        h = 1./(1.+np.exp(-z))
        return h

    def costFunction(self, theta, *args):
        x,y = args
        m = x.shape[0]
        #    print 'theta ',theta
        h = self.calcHypothesis(x, theta)
        #    print 'h ',h
        J = 1./m*(-y.T*np.log(h)-(1.-y.T)*np.log(1.-h))[0][0]
        #    print 'J ',J
        return J

    def costFunctionGrad(self, theta, *args):
        x,y = args
        m = x.shape[0]
        h = self.calcHypothesis(x, theta)
        grad = 1./m*x.T*(h-y)
        #    print 'grad ', grad
        return np.asarray(grad)[:,0] # return value should be a 1D array of size (ntheta,)

    def LogisticRegGradientDescent(self, x, y, alpha, niters):
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

    # x, y are of type matrix. theta is of type array
    def train(self, x,y, method='downhill', alpha=1.0, maximiter=100, tol=1e-03):
        col_one = np.asmatrix(np.ones(x.shape[0])).T
        x = np.c_[col_one, x]
        ntheta = x.shape[1]
        theta0 = np.zeros(ntheta)
        # choose the method
        if method == 'cg':
            theta = optimize.fmin_cg(self.costFunction, theta0, fprime=self.costFunctionGrad, args=(x,y), gtol=tol, maxiter=maximiter)
            #        theta = optimize.fmin_cg(costFunction, theta0, args=(x,y), gtol=tol, maxiter=maxiter)
            return theta
        elif method == 'bfgs':
            theta = optimize.fmin_bfgs(self.costFunction, theta0, fprime=self.costFunctionGrad, args=(x,y), gtol=tol, maxiter=maximiter)
            return theta
        elif method == 'downhill':
            theta = optimize.fmin(self.costFunction, theta0, args=(x,y), ftol=tol, maxiter = maximiter)
            return theta
        elif method == 'gd':
            theta, J_hist = self.LogisticRegGradientDescent(x, y, alpha, maximiter)
            return theta
    
    def predict(self,x):
        x = np.c_[1, x]
        if len(x) != len(self.theta):
            raise ValueError("length of theta and x are %d, %d" %(len(self.theta), len(x)))
        return self.calcHypothesis(x, self.theta)