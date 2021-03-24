import numpy as np
from cvxopt import matrix, solvers

class RR():
    """
    Implementation of ridge regression
    lambda_reg : regularization parameter
    """
    def __init__(self, lambda_reg):
        self.lambda_reg = lambda_reg
        
    #Fits X_train, assuming that Y_train contains values in {-1,1}
    def fit(self,X_train,Y_train):
        n,d = X_train.shape
        self.w_RR = np.linalg.solve(np.dot(X_train.T,X_train) + self.lambda_reg*n*np.eye(d),np.dot(X_train.T,Y_train))
        
    #Predicts Y_val {-1,1}
    def predict(self,X_val):
        Y_train_predict = np.dot(X_val,self.w_RR)
        Y_train_predict[Y_train_predict<0] = -1
        Y_train_predict[Y_train_predict>0] = 1
        return Y_train_predict  



class Kernel_RR():
    """
    Implementation of kernel ridge regression
    lambda_reg : regularization parameter
    """
    def __init__(self, lambda_reg):
        self.lambda_reg = lambda_reg
    
    def fit(self,K_train,Y_train):
        n = K_train.shape[0]
        alpha = np.linalg.solve(K_train + self.lambda_reg*n*np.eye(n),Y_train)
        return alpha

    
class SVM():
    """
    Implementation of SVM
    lambda_reg : regularization parameter
    """ 
    
    def fit(K_train,Y_train,lambda_reg):
        
        n = K_train.shape[0]

        p = -2*Y_train
        G = np.vstack([-np.diagflat(Y_train),np.diagflat(Y_train)])
        h1 = np.zeros(Y_train.shape)
        h2 = 1/(2*n*lambda_reg)*np.ones(Y_train.shape)
        h = np.vstack([h1,h2])
        Q = 2*K_train

        Q = matrix(Q)
        q = matrix(p)
        G = matrix(G)
        h = matrix(h)
        
        sol=solvers.qp(Q, q, G, h)
        
        return np.array(sol['x'])  





