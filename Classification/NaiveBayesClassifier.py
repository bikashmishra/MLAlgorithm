import numpy as np
from scipy import misc, sparse
from itertools import izip
import abc

class BaseNaiveBayes():
    """ 
    Base class for Naive Bayes classfier
    Two classifiers will be derived from this
    1. Gaussian Naive Bayes
    2. Discrete Naive Bayes for text processing (eg spam filter)
    """
    __metaclass__ = abc.ABCMeta
     
    @abc.abstractmethod
    def log_likelihood(self, X):
        """
        Input:
            X: array_like [nexamples] x [nfeatures]
            output: array_like [nexamples] x [nlabels]
        """
    def predict(self,X):
        """
        Input:
            X: array_like [nexamples] x [nfeatures]
        Output:
            array_like [nexamples] with entry being label
        """
        class_prob = self.log_likelihood(X)
        return self.classes_[np.argmax(class_prob, axis=1)]
        
    def predict_logprob(self, X):
        """
        Input:
            X: array_like [nexamples] x [nfeatures]
        Output:
            array_like [nexamples] x [nclasses]
        """    
        """ Calculate p(f1,f2,..,fn)"""
        class_prob = self.log_likelihood(X)
        log_total_prob = misc.logsumexp(class_prob, axis=1)
        return class_prob - log_total_prob
  
class GaussianNB(BaseNaiveBayes):
    """ 
    Gaussian Naive bayes
    Input for training: 
      X : array_like [nexamples] x [nfeatures]
      y: array_like [nexamples] holding labels    
    """
    
    def train(self, X, y):
        self.classes_ = np.unique(y).tolist()
        nexamples, nfeatures = X.shape
        nclasses = len(self.classes_)
        self.priors = np.zeros(nclasses)
        self.mu = np.zeros((nclasses,nfeatures))
        self.sig = np.zeros((nclasses,nfeatures))
        for i,yi in enumerate(self.classes_):
            self.priors[i] = np.float(np.sum(y == yi, axis=0))/nexamples
            self.mu[i,:] = np.mean(X[y == yi,:], axis=0)
            self.sig[i,:]= np.var(X[y == yi,:], axis=0)
                    
    def log_likelihood(self, X):
        """
        Input:
            X: array_like [nexamples] x [nfeatures]
            output: array_like [nexamples] x [nlabels]
        """
        X = np.atleast_2d(X)
        nexamples, nfeatures = X.shape
        nlabels = len(self.classes_)
        out = np.zeros((nexamples, nlabels))
        for j in range(nlabels):
            l_ij = -0.5*np.sum(np.log(2.0*np.pi*self.sig[j,:]))
            l_ij -= 0.5*np.sum((X-self.mu[j,:])**2/self.sig[j,:], axis=1)
            l_ij += np.log(self.priors[j])
            out[:,j] = l_ij
        return out      


class MultinomialNBTextClassifier(BaseNaiveBayes):
    """
    This classifier takes in documents in the form
    ['D1', 'D2',...] and resp. classes [C1, C2,..]
    as arrays_like
    where D is the document / bag of words
    
    The classifier can be trained in parts
    Hence, all the classes need to be known apriori
    """
    
    def __init__(self, classes, alpha=1.0):
        """ 
        alpha is the smoothing value
        classes is of type array_like
        """
        self.alpha_ = alpha
        self.classes_ = classes.tolist()
        self.class_counts_ = self.alpha_*np.ones(len(self.classes_))
        self.features_ = []

    def add_feature(self, f):
        self.features_.append(f)
    
    def train(self, docs, labels):
        """
        docs is array_like
        labels is array_like of same len as docs
        """
        if docs.size != labels.size:
            raise ValueError('Docs and labels sizes are mismatched %d vs %d' %(docs.size, labels.size))
        
        for entry in docs:
            for f in entry.split():
                if f not in self.features_:
                    self.add_feature(f)
                
#        self.feature_counts_ = sparse.csr_matrix((nclasses, nfeatures), dtype=np.float)
        nfeatures = len(self.features_)
        nclasses = len(self.classes_)
 #       feature_counts = sparse.lil_matrix((nclasses, nfeatures), dtype=np.float)
        self.feature_counts_ = np.zeros((nclasses, nfeatures), dtype=np.float)
                            
        self.count_class(labels)
        self.update_class_log_prior(self.class_counts_)

        self.count_feature(docs, labels)
        self.update_feature_prob(self.feature_counts_)
        
    def count_class(self, labels):
      #  class_list = self.classes_.tolist()
        for c in labels:
            self.class_counts_[self.classes_.index(c)] += 1
        
    def update_class_log_prior(self, class_count=None):
        if class_count is not None:
            self.class_log_prior_ = np.log(class_count) - np.log(sum(class_count))
             
    def count_feature(self, docs, labels):
        for entry, c in izip(docs, labels):
            class_index = self.classes_.index(c)
            for f in entry.split():
                self.feature_counts_[class_index][ self.features_.index(f)] += 1
                
 #       self.feature_counts_ = feature_counts.tocsr()
                
    def update_feature_prob(self, feature_count=None):
        """ feature_count is of type csr"""
        feature_count_smooth = feature_count + self.alpha_
        class_count_smooth = feature_count_smooth.sum(axis=1)
        self.feature_log_likelihood_ = np.log(feature_count_smooth) - np.log(class_count_smooth.reshape(-1,1))
        
    def log_likelihood(self, X):
        """
        Input:
            X: array_like or sparse matrix [nexamples] x [nfeatures]
            output: array_like [nexamples] x [nlabels]
        """
        _,nfeatures = X.shape
        
        if nfeatures != len(self.features_):
            raise ValueError("Expectign input with %d features" % len(self.features_))
        out = (X*self.feature_log_likelihood_.T + self.class_log_prior_)
        return out
    
    def predict_doc(self, docs, predict_log_p=False):
        """ 
        Take in a list of docs and cerate a feature array
        of size [nexamples]x[nfeatures]. This can be a sparse matrix
        This matrix/array is then sent to predict and log_likelihood
        """
        nfeatures = len(self.features_)
        nexamples = len(docs)
        X = sparse.csr_matrix((nexamples,nfeatures), dtype=np.float)
        
        iexample = 0
        for entry in docs:
            for f in entry.split():
                X[iexample,self.features_.index(f)] += 1
                
        if not predict_log_p:
            return self.predict(X)
        else:
            return self.predict_logprob(X)
        