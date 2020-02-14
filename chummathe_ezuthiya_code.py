from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from extract_vocab import load_word_emb
from torch.utils.data import Dataset,DataLoader
from net_utils import run_lstm
from model import Model
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from utils import train_model,test_model



N_word = 50 
batch_size =10
hidden_dim = 100

word_embed = load_word_emb('glove/glove.6B.50d.txt')



#train , valid   = SQLDataset('train') , SQLDataset('dev')
#train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)

#valid_dataloader = DataLoader(valid,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)



test = SQLDataset('test')
test_dataloader = DataLoader(test,batch_size = batch_size, shuffle=True, num_workers=1,collate_fn=collate_fn)


#g=next(iter(train_dataloader))
#print(g)



word_emb = WordEmbedding(N_word,word_embed)



#embeddings,length = word_emb.gen_x_batch(g['question_tokens'],g['column_headers'])
#print(embeddings)
#sql_query = np.asarray(g['sql_query'])
#agg_tensor = torch.from_numpy(sql_query)
#print(agg_tensor.shape)

#name_inp_var , name_len , col_len = word_emb.gen_column_batch( g['column_headers'])

#print(g['column_headers'] )
#print(name_inp_var.shape)
#print(name_len)
#print(col_len)


#rnn = nn.LSTM(N_word,hidden_dim,batch_first=True)
#ret_var,col_len = column_encode( rnn , name_inp_var,name_len,col_len )
#print(ret_var)



#model = Model(hidden_dim,N_word,word_emb)



#agg_model = AggPredictor(N_word,hidden_dim)
epochs = 5

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=0.01)

#train_model(model,epochs,optimizer,train_dataloader,valid_dataloader)




testmodel = Model(hidden_dim,N_word,word_emb)

testmodel.load_state_dict( torch.load('saved_models/agg_model.pth') )


test_model( testmodel,test_dataloader)




#question = [ 'What is the total salary of employee 3'.split(' ') ,  'What is the total salary of employee 3'.split(' ')  ] 
#columns =[ [ ['id'],['batch'],['name']]  ,   [ ['id'],['batch'],['name']]  ]
#scores = test_model( question, columns , (True,None,None) )

#print( torch.exp(scores))


