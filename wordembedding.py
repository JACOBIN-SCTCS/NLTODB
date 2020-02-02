import torch 
import json
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):

    def __init__(self,N_word,word_emb):
        super(WordEmbedding,self).__init__()
        self.N_word = N_word
        self.word_emb = word_emb

    def gen_x_batch(self,q,col=None):

        batch_size = len(q)
        val_embs = [] 

        val_len = np.zeros(batch_size,dtype = np.int64)
             
        for i,(q_one,col_one) in enumerate(zip(q,col)):
        
            q_val = [ self.word_emb.get(x,np.zeros(self.N_word,dtype=np.float32))  for x in q_one ]

            val_embs.append( [np.zeros(self.N_word,dtype=np.float32)] + q_val + [np.zeros(self.N_word,dtype=np.float32)]  )

            val_len[i] = len(q_val) + 2 
        
        max_len = max(val_len)
        val_emb_array = np.zeros((batch_size,max_len,self.N_word),dtype=np.float32)

        for i in range(batch_size):
            for j in range(len(val_embs[i])):
                val_emb_array[i,j,:] = val_embs[i][j]
        
        input_tensor = torch.from_numpy(val_emb_array)
        input_tensor_var = Variable(input_tensor)
        return input_tensor_var,val_len

    
