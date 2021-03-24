#!/usr/bin/env python
# coding: utf-8

# In[3]:


##########################################
#Import packages
##########################################

import pandas as pd
import numpy as np
from cvxopt import matrix, solvers

from datahandler import datahandler
import predictors
from kernels import Kernel
from sklearn import datasets
import matplotlib.pyplot as plt


#########################################################################################
#DATASET 0
#########################################################################################

##########################################
#KERNEL 1
##########################################

#Load Train and embedding
path_data = 'dataset/Xtr0.csv'
path_label = 'dataset/Ytr0.csv'

train_dataset = datahandler(path_data, path_label,features_generated=False)

train_dataset.Y[train_dataset.Y==0]=-1
train_dataset.compute_vocabulary(6)
train_dataset.mismatch_embedding(6,1,train_dataset.vocab)


X_train0, Y_train = train_dataset.X_embedded,train_dataset.Y



kernel = Kernel(Kernel.dot_product())
K0 = kernel.kernel_matrix(X_train0)

#Load Test and embedding
path_data = 'dataset/Xte0.csv'
path_label = 'dataset/Ytr0.csv'

test_dataset = datahandler(path_data, path_label,features_generated=False)

test_dataset.mismatch_embedding(6,1,train_dataset.vocab)
X_test0 = test_dataset.X_embedded

##########################################
#KERNEL 2
##########################################
train_dataset.compute_vocabulary(7)
train_dataset.spectral_embedding(7,train_dataset.vocab)

train_dataset.Y[train_dataset.Y==0]=-1
X_train1, Y_train = train_dataset.X_embedded,train_dataset.Y

kernel = Kernel(Kernel.dot_product())
K1 = kernel.kernel_matrix(X_train1)

#Load Test and embedding
path_data = 'dataset/Xte0.csv'
path_label = 'dataset/Ytr0.csv'

test_dataset = datahandler(path_data, path_label,features_generated=False)

test_dataset.spectral_embedding(7,train_dataset.vocab)
X_test1 = test_dataset.X_embedded

##########################################
#TRAINING SVM
##########################################
K = K0+K1

#Computing solution
lambda_reg = 0.5

kernel = Kernel(Kernel.dot_product())
alpha = predictors.SVM.fit(K,Y_train,lambda_reg)

##########################################
#PREDICT
##########################################
Y_predict0 = kernel.predict_multi(X_train0, X_test0,alpha)
Y_predict1 = kernel.predict_multi(X_train1, X_test1,alpha)

Y = Y_predict0 + Y_predict1 
Y_0 = np.sign(Y).astype(int)


#########################################################################################
#DATASET 1
#########################################################################################

##########################################
#KERNEL 1
##########################################

#Load Train and embedding
path_data = 'dataset/Xtr1.csv'
path_label = 'dataset/Ytr1.csv'

train_dataset = datahandler(path_data, path_label,features_generated=False)

train_dataset.Y[train_dataset.Y==0]=-1
train_dataset.compute_vocabulary(6)
train_dataset.mismatch_embedding(6,1,train_dataset.vocab)


X_train0, Y_train = train_dataset.X_embedded,train_dataset.Y



kernel = Kernel(Kernel.dot_product())
K0 = kernel.kernel_matrix(X_train0)

#Load Test and embedding
path_data = 'dataset/Xte1.csv'
path_label = 'dataset/Ytr1.csv'

test_dataset = datahandler(path_data, path_label,features_generated=False)

test_dataset.mismatch_embedding(6,1,train_dataset.vocab)
X_test0 = test_dataset.X_embedded

##########################################
#KERNEL 2
##########################################
train_dataset.compute_vocabulary(6)
train_dataset.spectral_embedding(6,train_dataset.vocab)

train_dataset.Y[train_dataset.Y==0]=-1
X_train1, Y_train = train_dataset.X_embedded,train_dataset.Y

kernel = Kernel(Kernel.dot_product())
K1 = kernel.kernel_matrix(X_train1)

#Load Test and embedding
path_data = 'dataset/Xte1.csv'
path_label = 'dataset/Ytr1.csv'

test_dataset = datahandler(path_data, path_label,features_generated=False)

test_dataset.spectral_embedding(6,train_dataset.vocab)
X_test1 = test_dataset.X_embedded



##########################################
#TRAINING SVM
##########################################
K = K0+K1

#Computing solution
lambda_reg = 0.7

kernel = Kernel(Kernel.dot_product())
alpha = predictors.SVM.fit(K,Y_train,lambda_reg)

##########################################
#PREDICT
##########################################
Y_predict0 = kernel.predict_multi(X_train0, X_test0,alpha)
Y_predict1 = kernel.predict_multi(X_train1, X_test1,alpha)

Y = Y_predict0 + Y_predict1 
Y_1 = np.sign(Y).astype(int)



#########################################################################################
#DATASET 2
#########################################################################################

##########################################
#KERNEL 1
##########################################

#Load Train and embedding
path_data = 'dataset/Xtr2.csv'
path_label = 'dataset/Ytr2.csv'

train_dataset = datahandler(path_data, path_label,features_generated=False)

train_dataset.Y[train_dataset.Y==0]=-1
train_dataset.compute_vocabulary(6)
train_dataset.mismatch_embedding(6,1,train_dataset.vocab)


X_train0, Y_train = train_dataset.X_embedded,train_dataset.Y



kernel = Kernel(Kernel.dot_product())
K0 = kernel.kernel_matrix(X_train0)

#Load Test and embedding
path_data = 'dataset/Xte2.csv'
path_label = 'dataset/Ytr2.csv'

test_dataset = datahandler(path_data, path_label,features_generated=False)

test_dataset.mismatch_embedding(6,1,train_dataset.vocab)
X_test0 = test_dataset.X_embedded

##########################################
#KERNEL 2
##########################################
train_dataset.compute_vocabulary(5)
train_dataset.spectral_embedding(5,train_dataset.vocab)

train_dataset.Y[train_dataset.Y==0]=-1
X_train1, Y_train = train_dataset.X_embedded,train_dataset.Y

kernel = Kernel(Kernel.dot_product())
K1 = kernel.kernel_matrix(X_train1)

#Load Test and embedding
path_data = 'dataset/Xte2.csv'
path_label = 'dataset/Ytr2.csv'

test_dataset = datahandler(path_data, path_label,features_generated=False)

test_dataset.spectral_embedding(5,train_dataset.vocab)
X_test1 = test_dataset.X_embedded



##########################################
#TRAINING SVM
##########################################
K = K0+K1

#Computing solution
lambda_reg = 0.7

kernel = Kernel(Kernel.dot_product())
alpha = predictors.SVM.fit(K,Y_train,lambda_reg)

##########################################
#PREDICT
##########################################
Y_predict0 = kernel.predict_multi(X_train0, X_test0,alpha)
Y_predict1 = kernel.predict_multi(X_train1, X_test1,alpha)

Y = Y_predict0 + Y_predict1 
Y_2 = np.sign(Y).astype(int)


#Write submission
predictions = Y_0.tolist() + Y_1.tolist() + Y_2.tolist()

#Put it in a dataframe
columns = ["Id","Bound"]
results_df = pd.DataFrame(columns=columns)


for i, data in enumerate(predictions, 0):
    
    row = [i, '1' if data[0] > 0 else '0']
    row_df = pd.DataFrame([row], columns=columns)
    results_df = pd.concat([results_df, row_df])
    

#write the csv Yte.csv
results_df.to_csv('Yte.csv', index=False)

