from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from extract_vocab import load_word_emb
from torch.utils.data import Dataset,DataLoader
from model import Model

import torch
from utils import train_model
import torch.nn as nn
import torch.optim as optim



filename= 'glove/glove.6B.50d.txt'

checkpoint_name = 'saved_models/agg_model.pth'

N_word= 50
batch_size = 10
hidden_dim = 100
n_epochs = 5



train_entry = (True,None,None)

word_embed = load_word_emb(filename)


train , valid = SQLDataset('train') , SQLDataset('dev')

train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)

valid_dataloader = DataLoader(valid,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)


word_emb =  WordEmbedding(N_word,word_embed)


model = Model(hidden_dim,N_word,word_emb)
optimizer = optim.Adam(model.parameters(),lr=0.01)

train_model(model,n_epochs,optimizer,train_dataloader,valid_dataloader,train_entry,checkpoint_name)




