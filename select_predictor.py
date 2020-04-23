import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from net_utils import run_lstm, column_encode

class SelectionPredictor(nn.Module):
    


    def __init__(self,N_word,N_h,N_depth=1):
        
        super().__init__()
	
        
        self.select_lstm=nn.LSTM(N_word, hidden_size=int(N_h/2),num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
        
        self.select_att=nn.Linear(N_h,1)
        self.select_colname_enc = nn.LSTM(N_word, hidden_size=int(N_h/2),num_layers=N_depth, batch_first=True,dropout=0.3, bidirectional=True)
        self.select_out_K=nn.Linear(N_h,N_h)
        self.select_out_col=nn.Linear(N_h,N_h)
        
        self.select_out=nn.Sequential( 
			nn.Tanh(),  
			nn.Linear(N_h,1)
                        )



    def forward(self,x_emb,x_len,col_input,col_token_num,col_len,hidden=None):
        
        batch_size=len(x_emb)
        max_x_len=max(x_len)
        
        emb_col , _ = column_encode(self.select_colname_enc,col_input,col_token_num,col_len)
        
        hidden = None
        if not hidden:

            h_enc, _ =run_lstm(self.select_lstm, x_emb,x_len )
        
        else:

            h_enc , _ = run_lstm(self.select_lstm,x_emb,x_len,hidden)

        
        #to compute the attention score
        attn_value=self.select_att(h_enc).squeeze(2)
        for idx,num in enumerate(x_len):
            if num<max_x_len:
                attn_value[idx,num:]=-100
        
        attention=F.softmax(attn_value,1)
        
        
        
        
        K_select=(h_enc*attention.unsqueeze(2).expand_as(h_enc)).sum(1)
        K_select_expand=K_select.unsqueeze(1)
        select_score = self.select_out(self.select_out_K(K_select_expand) + self.select_out_col(emb_col)).squeeze(2)		
        max_col_num=max(col_len)
        
        
        
        for idx,num in enumerate(col_len):
            if num<max_col_num:
                select_score[idx,num:]= -100
        
        return select_score
		
