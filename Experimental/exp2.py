
import torch
import torch.nn as nn
from NLTODB.net_utils import run_lstm
import numpy as np

class SqlovaCondPredictor(nn.Module):
    
    def __init__(self,embed_dim,hidden_dim,dropout=0.3,num_layers=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.condition_number = 3
        self.maximum_conditions = 4
        
        # Layers for the purpose of predicting the number of conditions
        
        self.cond_num_name_enc = nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        self.cond_num_col_att = nn.Linear(hidden_dim,1)
        self.cond_num_lstm =    nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        
        self.cond_num_out = nn.Sequential( 
            
                             nn.Linear(hidden_dim,hidden_dim),
                             nn.Tanh(),
                             nn.Linear(hidden_dim,(self.maximum_conditions+1))
                             
                             )
        
        self.col2hid1 = nn.Linear(hidden_dim,2*hidden_dim)
        self.col2hid2 = nn.Linear(hidden_dim,2*hidden_dim)
        
        self.cond_num_att = nn.Linear(hidden_dim,1)
        
        
        #--------------------------------------------------------------------
        
        # Layers for predicting the columns to be included in each statement
        
        #---------------------------------------------------------------------
        
        self.cond_col_name_enc =    nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        
        self.cond_col_lstm =    nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        
        self.cond_col_att  = nn.Linear(hidden_dim,hidden_dim)
         
        self.cond_col_out_k = nn.Linear(hidden_dim,hidden_dim)
        self.cond_col_out_col = nn.Linear(hidden_dim,hidden_dim)
         
        self.cond_col_out = nn.Sequential( nn.Tanh() ,nn.Linear(2*hidden_dim,1))
         
        #--------------------------------------------------------------------
        
        # Portion for predicting the operator corresponding to each condition
        
        #-------------------------------------------------------------------
        
        self.cond_op_name_enc = nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        self.cond_op_lstm     = nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        
        self.cond_op_attn    = nn.Linear(hidden_dim,hidden_dim)
        self.cond_op_out_k   = nn.Linear(hidden_dim, hidden_dim)
        self.cond_op_out_col = nn.Linear(hidden_dim,hidden_dim)
        self.cond_op_out     = nn.Sequential(
                        
                                nn.Linear(2*hidden_dim,hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim,self.condition_number)
                                )
        
        
        
        #-------------------------------------------------------------------
        
        # Portion for predicting the strings in each where clause condition
        
        #-------------------------------------------------------------------
        
        self.cond_str_name_enc = nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                         )
        
        self.cond_str_lstm = nn.LSTM(embed_dim,int(hidden_dim/2),batch_first=True,
                                         num_layers = num_layers ,
                                         bidirectional=True,
                                         dropout = dropout,
                                         
                                     )
        
        self.cond_op_str_attn = nn.Linear(hidden_dim,hidden_dim)
        
        self.DC               =  nn.Linear(hidden_dim,hidden_dim)
        self.WCC              =  nn.Linear(hidden_dim,hidden_dim)
        self.VOP              =  nn.Linear(self.condition_number,hidden_dim)
        
        self.where_str_out    = nn.Sequential(
                                nn.Linear(4*hidden_dim,hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim,2)
                                )
        
        
        
        #----------------------------------------------------------------------
        
        
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        self.sigmoid      = nn.Sigmoid()
        
    def forward(self,q,q_len ,col,col_len,gt_cond=None,gt_where_op=None):
        
        batch_size = len(q)
        max_x_length = max(q_len)
        max_col_length= max(col_len)
        
        e_num_col, _  = run_lstm(self.cond_num_name_enc, col, col_len)
        col_num_att_val = self.cond_num_col_att(e_num_col).squeeze(2)
        
        # Give negative score to non existent columns
        for b in range(batch_size):
            if col_len[b] < max_col_length:
                col_num_att_val[b,col_len[b]:] = -10000
        
        num_col_att = self.softmax_dim1(col_num_att_val)
        
        k_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        
        cond_num_h1 = self.col2hid1(k_num_col).view(batch_size,-1, int(self.hidden_dim/2)).transpose(0, 1).contiguous()
        cond_num_h2 = self.col2hid2(k_num_col).view(batch_size,-1, int(self.hidden_dim/2)).transpose(0, 1).contiguous()

        h_num_enc , _ = run_lstm(self.cond_num_lstm,q,q_len,hidden=(cond_num_h1,cond_num_h2) )
        num_att_val   = self.cond_num_att(h_num_enc).squeeze(2)
        
        for b in range(batch_size):
            if q_len[b]< max_x_length:
                num_att_val[b,q_len[b]:] = -10000

        num_att = self.softmax_dim1(num_att_val)   
        
        k_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(h_num_enc)).sum(1)
        cond_num_score = self.cond_num_out(k_cond_num)
        
      
        # ----------- Columns to be included in the WHERE clause-------------------------------
        
        
        e_cond_col, _ = run_lstm(self.cond_col_name_enc, col, col_len)
        h_col_enc , _ = run_lstm(self.cond_col_lstm,q, q_len)
        
        col_att_val   = torch.bmm(e_cond_col,self.cond_col_att(h_col_enc).transpose(1,2))
        
        for b in range(batch_size):
            if q_len[b] < max_x_length:
                col_att_val[b,:,q_len[b]:] = -10000

        col_att = self.softmax_dim2(col_att_val)
        
        k_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3) ).sum(2)

        # CHANGE FROM SQLNET
        
        y_cond_col = torch.cat([self.cond_col_out_k(k_cond_col),self.cond_col_out_col(e_cond_col)],dim=2)
        
        cond_col_score = self.cond_col_out(y_cond_col).squeeze(2)
        
        for b in range(batch_size):
            if col_len[b]<max_col_length:
                cond_col_score[b,col_len[b]:] = -1e+10
                
                
        #-------------------Operators corresponding to each condition-------
                
                
        chosen_col_gt =[]
        
        if gt_cond is None:
            
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(),axis=1)
            col_scores = self.sigmoid(cond_col_score)
            
            col_scores = col_scores.data.cpu().numpy()
            chosen_col_gt = [ list(np.argsort(-col_scores[b])[:cond_nums[b]])   
                             
                              for b in range(len(cond_nums))
                            ]
        else:
            
            chosen_col_gt = gt_cond
            
        
        e_cond_col , _  = run_lstm( self.cond_op_name_enc , col , col_len)
        col_emb = []
        
        for b in range(batch_size):
            cur_col_emb = torch.stack( [e_cond_col[b,x] for x in chosen_col_gt[b]]
                    + [e_cond_col[b,0]]*( self.maximum_conditions - len(chosen_col_gt[b])
                ))
            
            col_emb.append(cur_col_emb)
            
        col_emb =torch.stack(col_emb)
        
        h_op_enc , _ = run_lstm(self.cond_op_lstm ,q, q_len)
        
        op_att_val = torch.matmul(self.cond_op_attn(h_op_enc).unsqueeze(1) 
                                  , col_emb.unsqueeze(3)).squeeze(3)
        
        for b in range(batch_size):
            if q_len[b]<max_x_length:
                op_att_val[b,:, q_len[b]:] = -10000 
        
        # change from SQLNET
        op_att = self.softmax_dim2(op_att_val)
        
        k_cond_op = (h_op_enc.unsqueeze(1)*op_att.unsqueeze(3)).sum(2)
        
        op_vec = torch.cat([self.cond_op_out_k(k_cond_op),self.cond_op_out_col(col_emb)],dim=2)
        
        cond_op_score = self.cond_op_out(op_vec)
        
        # ------------------------------------------------------------------
        
        # Strings corresponding to each WHERE clause (Entirely different from
                            
        #   SQLNET           )
        
        #-------------------------------------------------------------------
        
        
        h_str_enc , _ = run_lstm(self.cond_str_lstm,q,q_len)
        e_cond_col, _ = run_lstm(self.cond_str_name_enc,col,col_len)
        
        col_emb = []
        
        for b in range(batch_size):
            cur_col_emb = torch.stack( [e_cond_col[b,x] for x in chosen_col_gt[b]]
                    + [e_cond_col[b,0]]*( 4 - len(chosen_col_gt[b])
                ))
            
            col_emb.append(cur_col_emb)
            
        col_emb =torch.stack(col_emb)
        
        str_op_att_val =  torch.matmul(self.cond_op_str_attn(h_str_enc).unsqueeze(1),
                                       
                                       col_emb.unsqueeze(3)
                                       ).squeeze(3)
        
        for b in range(batch_size):
            if q_len[b]<max_x_length:
                str_op_att_val[b,:, q_len[b]:] = -10000
        
        str_op_att = self.softmax_dim2(str_op_att_val)
        
        # One context from the operations performed for Operator
        c_n = (h_str_enc.unsqueeze(1) * str_op_att.unsqueeze(3)).sum(2)
        
        where_ops = []
        if gt_where_op:
            
            where_ops = gt_where_op
            
        else:
            op_score = cond_op_score
            op_score = self.softmax_dim2(op_score)
            
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(),axis=1)
            for b in range(batch_size):
                current_cond = []
                for j in range(cond_nums[b]):
                    idx = int(torch.argmax(op_score[b,j]))
                    current_cond.append(idx)
                where_ops.append(current_cond)
        
        wenc_op =[]
        for b in range(batch_size):
            operations = torch.zeros(self.maximum_conditions ,self.condition_number)
            current_ops = where_ops[b]
            idx_scatter = [] 
            
            length_operations = len(current_ops)
            
            for i in range(self.maximum_conditions):
                if i < length_operations:
                    
                    current_op_idx = current_ops[i]
                    idx_scatter.append([int(current_op_idx)])
                else:
                    
                    idx_scatter.append([0])
            
            operations = operations.scatter(1,torch.tensor(idx_scatter),1)
            
            wenc_op.append(operations)
        
        wenc_op = torch.stack(wenc_op)
        
        vec = torch.cat([  self.WCC(c_n) , self.DC(col_emb) , self.VOP(wenc_op)  ],dim=2)
        
        
        vec1e = vec.unsqueeze(2).expand(-1,-1,max_x_length,-1)
        wenc_ne = h_str_enc.unsqueeze(1).expand(-1,self.maximum_conditions,-1,-1)
        
        vec2 = torch.cat([wenc_ne,vec1e], dim=3)
        
        cond_str_score = self.where_str_out(vec2)
        
        for b in range(batch_size):
            
            if q_len[b]<max_x_length:
                
                cond_str_score[b,:,q_len[b]:,:] = -1e+10
        
        
       
    
        return cond_num_score,cond_col_score,cond_op_score,cond_str_score
    
    
    
    
        
        
        
        
    