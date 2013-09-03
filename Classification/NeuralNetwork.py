import numpy as np
from scipy import optimize
from itertools import izip

class NeuralNetwork:
    """ Class implementing Neural Networks. This class trains and predicts """
      
    def __init__(self, num_hidden_layers, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.weights = [None]*(self.num_hidden_layers+1)
        self.theta_grad = [None]*(self.num_hidden_layers+1)
        self.activation = [None]*(self.num_hidden_layers+2)
        self.z = [None]*(self.num_hidden_layers+2)
        self.J = 0.0
        self.delta = [None]*(self.num_hidden_layers+2)
        
    def randInitializeWeights(self, neuron_out, neuron_in):
        """  
        theta = RANDINITIALIZEWEIGHTS(neuron_in, neuron_out) randomly initializes the weights
        of a layer with neuron_in incoming connections and neuron_out outgoing
        connections.
        Note that theta should be set to a matrix of size(neuron_out, neuron_in) 
        """
        theta = np.asmatrix(np.random.rand(neuron_out, neuron_in))
        return theta
    
    def sigmoid(self, z):
        return 1./(1.+np.exp(-z))
    
    def sigmoidGradient(self,z):
        return np.multiply(self.sigmoid(z),(1.-self.sigmoid(z)))
    
    def addbias(self, a):
        """ a is of type matrix nexamples x layersize"""
        col_one = np.asmatrix(np.ones(a.shape[0])).T
        a = np.c_[col_one, a]
        return a
    
    def train(self, datain, dataout, lamda=0):
        """ 
        input is of type array/matrix m x nfeatures 
        output is of type array/matrix m x 1      
        """
        self.x = np.asmatrix(datain) 
        self.nexamples = datain.shape[0]
        self.input_layer_size = datain.shape[1]
        self.y = np.asmatrix(dataout)
        self.ydict = {}
        label = 0
        for key in np.asarray(self.y).flatten():
            if key not in self.ydict:
                self.ydict[key] = label
                label += 1
        self.num_labels = label
        print self.ydict
        
        self.lamda = lamda
        self.nexamples = datain.shape[0]
        
        """ Initialize thetas """
        """ theta is returned as type matrix """
        theta = self.randInitializeWeights(self.hidden_layer_size, self.x.shape[1]+1) #theta_(nlabel x nlabel-1)
        self.weights[0] = theta
        for i in range(self.num_hidden_layers-1):
            theta = self.randInitializeWeights(self.hidden_layer_size, self.hidden_layer_size+1) #theta_(nlabel x nlabel-1)
            self.weights[i+1] = theta
        theta = self.randInitializeWeights(self.num_labels, self.hidden_layer_size+1) 
        self.weights[self.num_hidden_layers] = theta
 
        """ Calc cost function """
        self.costFunction()
        
        """ Unroll initial thetas"""
#        self.unrollMatList(self.weights)
        

    def costFunction(self): 
        """ Forward Propagate"""
        print 'Forward propagating...'
        self.forwardPropagate()
        print '...done'
        print 'Calculating cost function...' 
        h = self.hypothesis.T
        y = self.y
        nexamples = self.nexamples
        nlabels = self.num_labels
        self.ymatrix = np.matrix((nlabels, nexamples))
        for i in range(nexamples):
            ym = np.asmatrix(np.zeros(nlabels)) # ym_(1 x nlabels)
#            ym[0,self.ydict[y[i,0]]] = 1.0
            ym[0,y[i,0]-1] = 1.0
            hi = h[:,i] # hi_(nlabels x 1)
            self.J += (-ym*np.log(hi)-(1.0-ym)*np.log(1.0-hi)).item(0,0)
            if i == 0:
                self.ymatrix = ym.T
            else:
                self.ymatrix = np.column_stack((self.ymatrix,ym.T)) # ymatrix_(nlabels x nexamples)
        self.J *= 1.0/nexamples        
        self.regularize()
        print '...done'
        return self.J
    
    def regularize(self):
        reg = 0.0;
        for theta in self.weights:
            reg += np.sum(theta**2)                        
        
        self.J += self.lamda/2.0/self.nexamples*reg 
                    
    def costFunctionGrad(self):
        """ Backward Propagate"""
        self.backPropagate()
        theta_grad_vec = self.unrollMatList(self.theta_grad)  
        return np.asarray(theta_grad_vec)[:,0]
        
    def forwardPropagate(self):
        """
        a^0 = activation^0 = x
        z^1 =  theta^1*a^0 ; a^1 = activation^1 = h(z^1)
        z^i = theta^i*a^(i-1) ; a^i = h(z^i)
        """
        a = self.x
        self.activation[0] = a
        count = 1
        for theta in self.weights:
            a = self.addbias(a)
            z = theta*a.T #layersize x nexamples
            self.z[count] = z
            a = self.sigmoid(z.T)
            self.activation[count]=a
            count += 1
        """ h_(m x nlabels )"""
        self.hypothesis = a # nexamples x nlabels
        
    def backPropagate(self):
        """ compute delta_L """
        nL = self.num_hidden_layers+1 # 0 index 
        delta = self.hypothesis.T - self.ymatrix # delta_(nlabels x nexamples)
        self.delta[nL] = delta
#        for (z, thet) in izip(self.z.reverse(),self.theta.reverse()):
        for l in range((nL-1),0,-1):  
            z = self.z[l]
            thet = self.weights[l] 
            delta = np.multiply(thet.T*delta, np.vstack((np.ones(z.shape[1]),self.sigmoidGradient(z)))) # delta_(layersize+1,  examples)
            delta = delta[2:,:]
            self.delta[l] = delta  
            """ theta_grad = (nlayersize^(l+1) x [nlayersize^l + 1])"""
            self.theta_grad[l] = self.theta_grad[l] + self.delta[l+1]*self.addbias(self.activation[l])
        self.theta_grad[0] = self.theta_grad[0] + self.delta[1]*self.addbias(self.activation[0])
        """ Regularize """
        for (th_grad,theta) in izip(self.theta_grad,self.weights):
            th_grad = th_grad/self.nexamples
            th_grad[:,2:] += self.lamda/self.nexamples*theta[:,2:]
          
    def unrollMatrix(self,mat):
        """ Row major flattening """
        return mat.flatten()
    
    def unrollMatList(self,matlist):
        vec = np.ndarray()
        for mat in matlist:
            vec.append(self.unrollMatrix(mat))
        return vec
    
    def rollVec(self, vec, (r,c)):
        np.reshape(vec,(r,c))
    
    def computeNumericalGradient(self):
        eps = 1.0e-04
        orig_w = self.weights   
        for w in self.weights:
            w = w + eps
            Jp = self.costFunction()
        self.weights = orig_w
        for w in self.weights:
            w = w - eps
            Jm = self.costFunction()
        self.weights = orig_w

    