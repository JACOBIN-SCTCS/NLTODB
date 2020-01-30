import torch
import torch.nn as nn
import numpy as np


class WordEmbedding(nn.Module):

    def __init__(self,word_emb,n_word):

        super().__init()
        self.word_emb = word_emb
        self.n_word = n_word



    # q stands for the question words token list
    # col stands for list containing column names 
    
    def  gen_x_batch(self,q,col):
        B =len(q)           # Get the number of tokens in the question
        val_embs = []       # Array to store the embeddding corresponding to each word

        val_len = np.zeros(B, dtype=np.int64)  # store the length of each question


        for i , (one_q,one_col) in enumerate(zip(q,col)):



