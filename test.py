from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from extract_vocab import load_word_emb
from torch.utils.data import Dataset,DataLoader
from agg_predictor import AggPredictor
from net_utils import column_encode,run_lstm
from model import Model
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from utils import train_model



N_word = 50 
batch_size =10
hidden_dim = 100

word_embed = load_word_emb('glove/glove.6B.50d.txt')

train , valid  = SQLDataset('train') , SQLDataset('dev')
train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)

valid_dataloader = DataLoader(valid,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)



#g=next(iter(sql_dataloader))
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



model = Model(hidden_dim,N_word,word_emb)



#agg_model = AggPredictor(N_word,hidden_dim)
epochs = 5

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

train_model(model,epochs,optimizer,train_dataloader,valid_dataloader)



'''
#agg_model.train()
model.train()
for e in range(epochs):
    #hidden = (torch.zeros((1,batch_size,hidden_dim)), torch.zeros((1,batch_size,hidden_dim)) )

    epoch_loss = 0

    for data in  sql_dataloader:
        #hidden = tuple([each.data for each in hidden])
        #agg_model.zero_grad()
        
        model.zero_grad()
        optimizer.zero_grad()

        #embeddings ,length = word_emb.gen_x_batch(data['question_tokens'],data['column_headers'])
        #scores = agg_model(embeddings ,length,hidden)
        
        scores = model(data['question_tokens'] , data['column_headers'] ,(True,None,None) )

       
        
        #agg_tensor = torch.from_numpy( np.asarray(data['sql_query']) )

        #loss = criterion(scores,agg_tensor)
        
        loss = model.loss(scores,data['sql_query'])

        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()

        #print('Loss {} ----- {}'.format(e,loss.item()))
        epoch_loss += loss.item()
    
    print('Loss {} ----- {}'.format(e,  epoch_loss / len(sql_dataloader) ))

'''



