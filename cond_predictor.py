import torch
import torch.nn as nn
from net_utils import column_encode,run_lstm



class CondPredictor(nn.Module):

    def __init__(self, embed_dim,hidden_dim, num_layers=1):

        super().__init__()
        
        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim



        # LSTMS for the purpose of predicting the number of conditions
        #------------------------------------------------------------------

        self.cond_num_name_enc = nn.LSTM(embed_dim , int(hidden_dim/2),num_layers=num_layers,batch_first=True , bidirectional=True)
        self.cond_num_col_att  = nn.Linear(hidden_dim,1)
        self.cond_num_lstm  = nn.LSTM(embed_dim , int(hidden_dim/2), num_layers=2, batch_first=True,dropout=0.3,bidirectional=True )    
        self.cond_num_att  = nn.Linear(hidden_dim,1)

        # Limit set for the number of conditions are 5
        self.cond_num_out = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim,5)
                )
        
        #---------------------------------------------------------------



        
        self.col2hid1       = nn.Linear(hidden_dim,2*hidden_dim)
        self.col2hid2       = nn.Linear( hidden_dim , 2*hidden_dim )
        


        self.softmax = nn.Softmax(dim=1)

    def forward(self,q,q_len,col_inp_var, name_length,col_length):



        # Portion for prediciting the number of conditions in the WHERE clause.
        #______________________________________________________________________



        batch_size = len(q_len)
        max_x_len = max(q_len)

        e_num_col , col_length = column_encode( self.cond_num_name_enc,col_inp_var,name_length,col_length )

        col_num_att_val = self.cond_num_col_att(e_num_col).squeeze()
        
        for idx, num in enumerate(col_length):

            if num < max(col_length):

                col_num_att_val[idx, num:] =-100

        num_col_att = self.softmax(col_num_att_val )
        k_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)

        # Get the hidden states for the RNN
        
        cond_num_h1  = self.col2hid1(k_num_col).view(batch_size,-1,int(self.hidden_dim/2)).transpose(0,1).contiguous()
        cond_num_h2  = self.col2hid2(k_num_col).view(batch_size,-1,int(self.hidden_dim/2)).transpose(0,1).contiguous()

        
        h_num_enc, _  = run_lstm( self.cond_num_lstm, q,q_len,hidden=(cond_num_h1,cond_num_h2))
        num_att_val = self.cond_num_att(h_num_enc).squeeze()

        for i , num in enumerate(q_len ):
            if num < max_x_len:
                num_att_val[i,num:] = -100
        num_att = self.softmax(num_att_val)

        k_cond_num = ( h_num_enc * num_att.unsqueeze(2).expand_as(h_num_enc) ).sum(1)
        cond_num_score = self.cond_num_out(k_cond_num)
        cond_num_score = self.softmax(cond_num_score)

        return cond_num_score
       
