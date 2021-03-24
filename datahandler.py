import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from tqdm import tqdm

class datahandler():
    
    '''
    Loads a csv type data where data is an array separated by spaces
    Here the X and y data are supposed to exist in 2 different csv
    '''
    def load_data(self,path_data,path_label,features_generated = True):
        if features_generated:
            df_data = pd.read_csv(path_data, names = ['features'])
            df_data = pd.DataFrame(df_data.features.str.split(' ',99).tolist(),
                                             columns = ['x{}'.format(i) for i in range(100)])
            df_data = df_data.to_numpy().astype(float)

        else:
                
            df_data = pd.read_csv(path_data, header=0)
            df_data = df_data['seq'].to_numpy().reshape(-1,1)
        
        df_label = pd.read_csv(path_label, header=0)  
        df_label = df_label.Bound.to_numpy().astype(float).reshape(-1,1)
            
        return df_data, df_label

     
    def __init__(self, path_data,path_label,features_generated):
        self.X, self.Y = self.load_data(path_data,path_label,features_generated)
        self.vocab = {}
    
    
    
    '''
    Splits the data set into a train and validation dataset 
    with a ratio p
    '''    
    def train_val(self,X,Y,p=0.75, print_shape=False):
        Y[Y==0] = -1
        XY = np.hstack([X,Y])
        
        msk = np.random.rand(X.shape[0]) < p
        X_train, Y_train = XY[msk][:,:-1], XY[msk][:,-1]
        X_val, Y_val = XY[~msk][:,:-1], XY[~msk][:,-1]

        Y_train = Y_train.reshape(-1,1)
        Y_val = Y_val.reshape(-1,1)
        
        if print_shape:
            print('X_train and Y_train shape: {},{}'.format(X_train.shape, Y_train.shape))
            print('X_val and Y_val shape: {},{}'.format(X_val.shape, Y_val.shape))
        
        return X_train, X_val, Y_train, Y_val
    
       

    def compute_vocabulary(self,k, print_vocab = False):
        n = self.X.shape[0]
        d = len(self.X[0][0])
        self.vocab = {}

        idx = 0
        print("Computing vocabulary...")
        for x in self.X:
            x = x[0]
            for j in range(d-k+1):
                kmer = x[j:j+k]
                if kmer not in self.vocab:
                    self.vocab[kmer] =idx
                    idx+=1   
        if print_vocab:
            print(self.vocab)
            
    ##-------------------------------------------
    # Spectral embedding  
    ##-------------------------------------------
    def spectral_embedding(self,k,vocab):
        n = self.X.shape[0]
        d = len(self.X[0][0])
        vocab_size = len(vocab)
        
        embedding = [{} for x in self.X]
        
        print("Counting kmers...")
        
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[0][j: j + k]
                if kmer not in vocab:
                    pass
                else:
                    if kmer in embedding[i]:
                        embedding[i][kmer] += 1
                    else:
                        embedding[i][kmer] = 1
        
        self.X_embedded = np.zeros((self.X.shape[0],vocab_size))
        
        print("Computing dataset embedding...")
        for i in tqdm(range(n), disable=False):
            for kmer,value in embedding[i].items():
                
                self.X_embedded[i,vocab[kmer]] = value
                
    ##-------------------------------------------
    # Spectral embedding  
    ##-------------------------------------------
    def mismatch_count(self,kmer1,kmer2):
        return sum(c1!=c2 for c1,c2 in zip(kmer1,kmer2))
    
    def search_mismatches(self,kmer,m,vocab):
        return [elt for elt in vocab if self.mismatch_count(kmer,elt)<=m]
    
    def mismatch_listing(self,m,vocab):
        mismatch_dic = {}
        "Computing mismatches..."
        for kmer in tqdm(vocab, disable=False):
            mismatches = self.search_mismatches(kmer,m,vocab)
            mismatch_dic[kmer] = mismatches
        return mismatch_dic

    def mismatch_embedding(self,k,m,vocab):
        n = self.X.shape[0]
        d = len(self.X[0][0])
        vocab_size = len(vocab)
        
        embedding = [{} for x in self.X]
        
        mismatch_dic = self.mismatch_listing(m,vocab)
        
        print("Counting kmers up to {} mismatches ...".format(m))
        
        for i,x in enumerate(tqdm(self.X, disable=False)):
            for j in range(d - k + 1):
                kmer = x[0][j: j + k]
                if kmer not in mismatch_dic:
                    pass
                else:
                    mismatch_list =  mismatch_dic[kmer]
                
                    for mismatch in mismatch_list:
                        if mismatch in embedding[i]:
                            embedding[i][mismatch] += 1
                        else:
                            embedding[i][mismatch] = 1
        
        self.X_embedded = np.zeros((self.X.shape[0],vocab_size))
        
        print("Computing dataset embedding...")
        for i in tqdm(range(n), disable=False):
            for kmer,value in embedding[i].items():
                self.X_embedded[i,vocab[kmer]] = value
            
    
        
        

