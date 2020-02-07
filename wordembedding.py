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

 





    def gen_column_batch(self,cols):
        


        # Stores numbers of columns in the corresponding table to which each  question is related
        col_len  = np.zeros(len(cols), dtype=np.int64)  



        # create a single list containing all the columns in the batch
        names =[]

        for i, col in enumerate(cols):
            names = names + col   
            col_len[i] = len(col)
        

        name_inp_var,name_len = self.list_to_batch(names)
        
        return name_inp_var,name_len,col_len






    def list_to_batch(self,col_list):
        
        total_columns = len(col_list)

        val_embs = []
        val_len = np.zeros(total_columns,dtype=np.int64 )

        for i,col in enumerate(col_list):
            val = [ self.word_emb.get(x, np.zeros(self.N_word,dtype=np.float32)) for x in col  ] 

            val_embs.append(val)
            val_len[i] = len(val)

        max_len = max(val_len)

        val_emb_array = np.zeros( (total_columns,max_len,self.N_word) , dtype=np.float32  )

        for i in range(total_columns):
            for j in range( len(val_embs[i])  ):
                val_emb_array[i,j,:] = val_embs[i][j]

        

        val_inp = torch.from_numpy(val_emb_array)
        val_inp_var = Variable(val_inp)

        return val_inp_var , val_len




