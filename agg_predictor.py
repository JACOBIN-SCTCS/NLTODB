import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np  

class AggPredictor(nn.Module):

    def __init__(self, embed_dim,hidden_dim,n_layers=1,dropout=0.2,bidirectional=False):

        super().__init__()
        
        self.rnn = nn.LSTM(embed_dim,hidden_dim,n_layers,batch_first=True,dropout=dropout,bidirectional=bidirectional)

        self.attn = nn.Linear(hidden_dim,1)
       
        self.softmax = nn.Softmax(dim=1)

        self.agg_out = nn.Sequential(  
                        nn.Linear(hidden_dim,hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim,6),
                )



    def forward(self,q,hidden):

        output,hidden = self.rnn(q,hidden)
        att_val = self.attn(output).squeeze()

        att = F.softmax(att_val,dim=1)
        k_agg = (output * att.unsqueeze(2).expand_as(output)).sum(1)
        agg_score = self.agg_out(k_agg)
        return agg_score

'''
agg_predictor = AggPredictor(50,100)
q = torch.randn(64,15,50)
hidden = (torch.randn(1,64,100), torch.randn(1,64,100))
score = agg_predictor(q,hidden)
print(score)
'''
