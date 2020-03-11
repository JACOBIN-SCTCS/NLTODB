import torch
import torch.nn as nn
from net_utils import column_encode


class CondPredictor(nn.Module):

    def __init__(self, embed_dim,hidden_dim, num_layers=1):

        super().__init__()
        
        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim

        self.cond_num_name_enc = nn.LSTM(embed_dim , int(hidden_dim/2),num_layers=num_layers,batch_first=True , bidirectional=True)
        self.cond_num_col_att  = nn.Linear(hidden_dim,1)
            


        self.softmax = nn.Softmax(dim=1)

    def forward(self,q,q_len,col_inp_var, name_length,col_length):



        # Portion for prediciting the number of conditions in the WHERE clause.
        #______________________________________________________________________



        e_num_col , col_length = column_encode( self.cond_num_name_enc,col_inp_var,name_length,col_length )

        col_num_att_val = self.cond_num_col_att(e_num_col).squeeze()
        
        for idx, num in enumerate(col_length):

            if num < max(col_length):

                col_num_att_val[idx, num:] =-100

        num_col_att = self.softmax(col_num_att_val )
        
        
