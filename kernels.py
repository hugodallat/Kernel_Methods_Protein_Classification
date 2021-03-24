import numpy as np
from cvxopt import matrix, solvers
from tqdm import tqdm

class Kernel():
    
    """
    Kernel class allowing the computing of the gram matrix according to various kernel
    If applied to string data, this kernel should be chosen in accordance with the 
    way the datahandler generated the features vector.
    """
    def dot_product():
        return lambda x,y : np.dot((x.reshape(-1,1)).T,y.reshape(-1,1))[0,0]
    
    def gaussian(sigma):
        return lambda x, y : 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-np.linalg.norm(x - y)**2/(2*sigma**2))
    
    def __init__(self,func):
        self.kernel = func
    
    #Creates kernel matrix according to the kernel chosen.
    def kernel_matrix(self,X):
        n,d = X.shape
        K = np.zeros((n,n))
        print('Computing kernel matrix...')
        for i in tqdm(range(n), disable=False):
            for j in range(i+1):
                x_i = X[i,:]
                x_j = X[j,:]
                K[i,j] = self.kernel(x_i,x_j)
                K[j,i] = K[i,j]
        return K
    
    #Predicts the label of new data.
    #WARNING : If evaluating the label of the training data, simply do
    #the dot product between the gram matrix K_train and alpha
    def predict(self,X_train, X,alpha):
        n_train = X_train.shape[0]
        n_val = X.shape[0]     
        K_predict = np.zeros((n_val,n_train))
        print('Predicting...')
        for i in tqdm(range(n_val), disable=False):
            for j in range(n_train):
                x_j = X_train[j,:]
                x_i = X[i,:]
                K_predict[i,j] = self.kernel(x_i,x_j)
        
        Y_train_predict = np.dot(K_predict,alpha)
        Y_train_predict[Y_train_predict<0] = -1
        Y_train_predict[Y_train_predict>0] = 1
        return Y_train_predict
    
    def predict_multi(self,X_train, X,alpha):
        n_train = X_train.shape[0]
        n_val = X.shape[0]     
        K_predict = np.zeros((n_val,n_train))
        print('Predicting...')
        for i in tqdm(range(n_val), disable=False):
            for j in range(n_train):
                x_j = X_train[j,:]
                x_i = X[i,:]
                K_predict[i,j] = self.kernel(x_i,x_j)
        
        Y_train_predict = np.dot(K_predict,alpha)
        return Y_train_predict

