from wordembedding import WordEmbedding
from model import Model
from extract_vocab import load_word_emb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import gen_query_acc,gen_sql_query
import sys
from word_mapping import *

filename= 'glove/glove.42B.300d.txt'
agg_checkpoint_name = 'saved_models/agg_predictor.pth'
select_checkpoint_name = 'saved_models/sel_predictor.pth'
cond_checkpoint_name = 'saved_models/cond_predictor.pth'


N_word= 300
batch_size = 10
hidden_dim = 100
n_epochs = 5
table_name = 'EMPLOYEE'


#word_embed = load_word_emb(filename)

word_emb =  WordEmbedding(N_word)


model = Model(hidden_dim,N_word,word_emb)
model.agg_predictor.load_state_dict( torch.load(agg_checkpoint_name) )
model.cond_predictor.load_state_dict(torch.load(cond_checkpoint_name))
model.sel_predictor.load_state_dict(torch.load(select_checkpoint_name))

model.eval()

sentence=sys.argv[1]
sentence = process_sentence(sentence)

question = [ sentence.split(' ') ,  sentence.split(' ')  ] 


columns =[ [ ['id'],['batch'],['name'],['salary']]  ,   [ ['id'],['batch'],['name'],['salary']]  ]

scores = model( question, columns , (True,True,True) )

agg = torch.argmax(torch.exp(scores[0]),dim=1)
sel = torch.argmax(torch.exp(scores[1]),dim=1)
where_clause_query = gen_query_acc(scores[2], question  )
for i in range( len(agg) -1 ):

    query = gen_sql_query(agg[i],sel[i],where_clause_query[i],columns[i],table_name)
    print(query)
    

    
    
