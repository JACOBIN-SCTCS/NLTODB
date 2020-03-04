from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from torch.utils.data import Dataset,DataLoader
from model import Model
from extract_vocab import load_word_emb
import torch
import numpy as np
from utils import test_model
import torch.nn as nn
import torch.optim as optim



filename= 'glove/glove.6B.50d.txt'
checkpoint_name = 'saved_models/agg_model.pth'


N_word= 50
batch_size = 10
hidden_dim = 100
n_epochs = 5



word_embed = load_word_emb(filename)


test =  SQLDataset('test')
test_loader = DataLoader(test,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)


word_emb =  WordEmbedding(N_word,word_embed)


model = Model(hidden_dim,N_word,word_emb)
model.load_state_dict( torch.load(checkpoint_name) )


optimizer = optim.Adam(model.parameters(),lr=0.01)

test_model(model,test_loader )

