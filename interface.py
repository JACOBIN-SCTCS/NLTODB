from wordembedding import WordEmbedding
from model import Model
from extract_vocab import load_word_emb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import gen_query_acc



filename= 'glove/glove.6B.50d.txt'
agg_checkpoint_name = 'saved_models/agg_predictor.pth'
select_checkpoint_name = None
cond_checkpoint_name = 'saved_models/cond_predictor.pth'


N_word= 50
batch_size = 10
hidden_dim = 100
n_epochs = 5



word_embed = load_word_emb(filename)

word_emb =  WordEmbedding(N_word,word_embed)


model = Model(hidden_dim,N_word,word_emb)
model.agg_predictor.load_state_dict( torch.load(agg_checkpoint_name) )
model.cond_predictor.load_state_dict(torch.load(cond_checkpoint_name))


model.eval()

question = [ 'What is the salary of employee having  id 3'.split(' ') ,  'What is the total salary of employee 3'.split(' ')  ] 


columns =[ [ ['id'],['batch'],['name'],['salary']]  ,   [ ['id'],['batch'],['name'],['salary']]  ]

scores = model( question, columns , (True,None,True) )

out = torch.argmax(torch.exp(scores[0]),dim=1)
where_clause_query = gen_query_acc(scores[2], question  )
for i in range( len(out) -1 ):
    print(model.agg_ops[out[i]])
    print(where_clause_query[i])
    

    
    
