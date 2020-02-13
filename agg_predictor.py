import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np  
from net_utils import run_lstm



''' 

        Neural Network for aggregator predictor 
        Implementation from Seq2sql paper
        
        Aggregator prediction is viewed as  a classification problem  with output being probability distribution over 6 aggregator operations which are
            [ '' , SUM , AVG, MIN , MAX , AVG ]

        Steps needed to calculate aggregation operator without giving attention to column is given below
        _________________________________________________________________________________________________________

            1)   Run an LSTM Network over the input sequence collecting the output from each token in the sequence at each time step
            2)   Calculate attention score for each token 
            3)   Multiply the output from each token with their corresponding scores
            4)   Add all the outputs into a single output vector 
            5)   Pass through a fully connected layer with Tanh activation function followed by a fully connected layer having 6 output dimensions to obtain score



'''


class AggPredictor(nn.Module):

    def __init__(self, embed_dim,hidden_dim,n_layers=1):

        super().__init__()
        
        self.rnn = nn.LSTM(embed_dim,int(hidden_dim/2),n_layers,batch_first=True,bidirectional=True)

        self.attn = nn.Linear(hidden_dim,1)
       
        self.softmax = nn.Softmax(dim=1)

        self.agg_out = nn.Sequential(  
                        nn.Linear(hidden_dim,hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim,6),
                )



    def forward(self,q,q_len,hidden):

        max_q_len = max(q_len)              # For the purpose of padding upto length of the largest question 

        output,hidden =  run_lstm(self.rnn,q,q_len,hidden)
        att_val = self.attn(output).squeeze()

        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val[idx,num:] = -100            # Give attention value -100 to words that do not belong to question

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
