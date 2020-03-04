from wordembedding import WordEmbedding
from model import Model
from extract_vocab import load_word_emb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim



filename= 'glove/glove.6B.50d.txt'
checkpoint_name = 'saved_models/agg_model.pth'


N_word= 50
batch_size = 10
hidden_dim = 100
n_epochs = 5



word_embed = load_word_emb(filename)

word_emb =  WordEmbedding(N_word,word_embed)


model = Model(hidden_dim,N_word,word_emb)
model.load_state_dict( torch.load(checkpoint_name) )

question = [ 'What is the total salary of employee 3'.split(' ') ,  'What is the total salary of employee 3'.split(' ')  ] 


columns =[ [ ['id'],['batch'],['name']]  ,   [ ['id'],['batch'],['name']]  ]

scores = model( question, columns , (True,None,None) )

out = torch.argmax(torch.exp(scores[0]),dim=1)
for i in range( len(out) -1 ):
    print(model.agg_ops[out[i]])

