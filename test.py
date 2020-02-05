from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from extract_vocab import load_word_emb
from torch.utils.data import Dataset,DataLoader
from agg_predictor import AggPredictor
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np



N_word = 50 
batch_size =10
hidden_dim = 100

word_embed = load_word_emb('glove/glove.6B.50d.txt')

sq = SQLDataset('train')
sql_dataloader = DataLoader(sq,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)
g=next(iter(sql_dataloader))

word_emb = WordEmbedding(N_word,word_embed)
#embeddings,length = word_emb.gen_x_batch(g['question_tokens'],g['column_headers'])
#print(embeddings)
#sql_query = np.asarray(g['sql_query'])
#agg_tensor = torch.from_numpy(sql_query)
#print(agg_tensor.shape)


agg_model = AggPredictor(N_word,hidden_dim)
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(agg_model.parameters(),lr=0.01)


agg_model.train()
for e in range(epochs):
    hidden = (torch.zeros((1,batch_size,hidden_dim)), torch.zeros((1,batch_size,hidden_dim)) )

    for data in  sql_dataloader:
        hidden = tuple([each.data for each in hidden])
        agg_model.zero_grad()

        embeddings ,length = word_emb.gen_x_batch(data['question_tokens'],data['column_headers'])
        scores = agg_model(embeddings ,length,hidden)
            
        agg_tensor = torch.from_numpy( np.asarray(data['sql_query']) )

        loss = criterion(scores,agg_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(agg_model.parameters(),5)
        optimizer.step()

        print('Loss {} ----- {}'.format(e,loss.item()))
        
        

