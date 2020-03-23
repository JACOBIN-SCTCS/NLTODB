import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from net_utils import run_lstm, column_encode

class SelectionPredictor(nn.Module):
	def __init__(self,N_word,N_h,N_depth,max_tok_num):
		super().__init__()
		self.max_tok_num=max_tok_num
		self.select_lstm=nn.LSTM(N_word, hidden_size=N_h/2,no_layers=N_depth, batch_first=True, dropout=0.3, bidirection=True)

		self.select_att=nn.Linear(N_h,1)
		self.select_colname_enc = nn.LSTM(N_word, hidden_size=N_h/2,no_layers=N_depth, batch_first=True,dropout=0.3, bidirectional=True)
      
		self.select_out_K=nn.Linear(N_h,N_h)
		
		self.select_out_col=nn.Linear(N_h,N_h)
		
		self.select_out=nn.Sequential( 
				nn.Tanh(),  
				nn.Linear(N_h,1))


	def forward(self,x_emb,x_len,col_input,col_token_num,col_len,col_num):

		batch_size=len(x_emb)
		max_x_len=max(x_len)


		emb_col= column_encode(self.select_colname_enc,col_input,col_token_num,col_len)


		h_enc=run_lstm(self.select_lstm, x_emb,x_len)

		#to compute the attention score

		attn_value=self.select_att(h_enc).squeeze();

		for idx,num in enumerate(x_len):

			if num<max_x_len:
				attn_value[idx,num:]=-100

		attention=F.softmax(attn_value,1)


		K_select=(h_enc*att.unsqueeze(2).expand_as(h_enc)).sum(1)

		K_select_expand=K_sel.usqueeze(1)

		
		select_score = self.select_out(self.select_out_K(K_select_expand) + self.select_out_col(emb_col)).squeeze()		
		
		max_col_num=max(col_num)

		for idx,num in enumerate(col_num):

			if num<max_col_num:
				select_score[idx,num:]= -100

		return select_score
		
