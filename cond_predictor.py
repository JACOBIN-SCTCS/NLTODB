import torch
import torch.nn as nn
from net_utils import column_encode,run_lstm
import numpy as np
from torch.autograd import Variable


class CondPredictor(nn.Module):

    def __init__(self, embed_dim,hidden_dim, num_layers=2,max_tok_num=200 , dropout=0.3):

        super().__init__()
        
        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim
        self.max_tok_num = max_tok_num


        # Layers for the purpose of predicting the number of conditions
        #------------------------------------------------------------------

        self.cond_num_name_enc = nn.LSTM(embed_dim , int(hidden_dim/2),num_layers=num_layers,batch_first=True ,dropout=dropout, bidirectional=True)
        self.cond_num_col_att  = nn.Linear(hidden_dim,1)
        self.cond_num_lstm  = nn.LSTM(embed_dim , int(hidden_dim/2), num_layers=2, batch_first=True,dropout=dropout,bidirectional=True )    
        self.cond_num_att  = nn.Linear(hidden_dim,1)

        # Limit set for the number of conditions are 5
        self.cond_num_out = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim,5)
                )
        
        #---------------------------------------------------------------

        # Layers for the prediction of the columns to be included in the WHERE clause.

        #---------------------------------------------------------------
        

        self.cond_col_name_enc = nn.LSTM(embed_dim , int(hidden_dim/2),num_layers=num_layers,batch_first=True,dropout=dropout,bidirectional=True)
        self.cond_col_lstm = nn.LSTM(embed_dim , int(hidden_dim/2),num_layers=num_layers,batch_first=True,dropout=dropout,bidirectional=True)

        self.cond_col_att = nn.Linear(hidden_dim,hidden_dim)

        
        self.cond_col_out_k = nn.Linear(hidden_dim,hidden_dim)
        self.cond_col_out_col = nn.Linear(hidden_dim,hidden_dim)
        self.cond_col_out = nn.Sequential( nn.Tanh(), nn.Linear(hidden_dim,1) )
        
        #---------------------------------------------------------------

        # Layers for the predicition of the operator corresponding to each column predicted

        #---------------------------------------------------------------

        
        self.cond_op_name_enc = nn.LSTM(embed_dim, int(hidden_dim/2), num_layers = num_layers,batch_first=True,dropout=dropout,bidirectional=True)
        self.cond_op_lstm     = nn.LSTM(embed_dim, int(hidden_dim/2), num_layers = num_layers,batch_first=True,dropout=dropout,bidirectional=True)
        self.cond_op_att      = nn.Linear(hidden_dim, hidden_dim)
        self.cond_op_out_k    = nn.Linear(hidden_dim, hidden_dim)
        self.cond_op_out_col  = nn.Linear(hidden_dim, hidden_dim)

        self.cond_op_out      = nn.Sequential( nn.Linear(hidden_dim , hidden_dim ), nn.Tanh()  , nn.Linear(hidden_dim,3))

        
        #---------------------------------------------------------------

        # Layers for the prediction of the values corresponding to each WHERE clause

        #---------------------------------------------------------------


        self.cond_str_lstm = nn.LSTM(embed_dim,int(hidden_dim/2) ,num_layers=num_layers,batch_first=True, dropout=dropout,bidirectional=True )

        self.cond_str_decoder = nn.LSTM(max_tok_num, hidden_dim,num_layers=num_layers,
                    batch_first=True,dropout=0.3
                )

        self.cond_str_name_enc = nn.LSTM(embed_dim,int(hidden_dim/2) ,num_layers=num_layers,batch_first=True,dropout=dropout, bidirectional=True )


        self.cond_str_out_g = nn.Linear(hidden_dim,hidden_dim)
        self.cond_str_out_h = nn.Linear(hidden_dim,hidden_dim)
        self.cond_str_out_col = nn.Linear(hidden_dim,hidden_dim)

        self.cond_str_out = nn.Sequential( nn.Tanh(), nn.Linear(hidden_dim,1) )


        #--------------------------------------------------------------


        self.col2hid1       = nn.Linear(hidden_dim,2*hidden_dim)
        self.col2hid2       = nn.Linear( hidden_dim , 2*hidden_dim )
        


        self.softmax = nn.Softmax(dim=1)

    
    # Used for supporting the training process by supplying the  condition 
    # strings corresponding to each query during training

    def gen_gt_batch(self,gt_where):

        batch_size = len(gt_where)
        max_len =  max( [  max(  [ len(tok) for tok in tok_seq] +[0] )

                    for tok_seq in gt_where]) - 1

        if max_len < 1:
            max_len = 1

        ret_array = np.zeros((batch_size,4,max_len,self.max_tok_num), dtype=np.float32 )

        ret_len = np.zeros((batch_size , 4))

        for b,tok_seq in enumerate(gt_where):

            idx =0
            for idx,one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1] # Get everything except thee last
                ret_len[b,idx] = len(out_one_tok_seq)
                for t , tok_id in enumerate(out_one_tok_seq):

                    ret_array[b,idx,t,tok_id] =1 

            if idx< 3:
                ret_array[b,idx+1: , 0,1] = 1
                ret_len[b,idx+1:]         = 1

        ret_inp = torch.from_numpy(ret_array)
        # cuda ret_inp down here

        ret_inp_var = Variable(ret_inp)

        return ret_inp_var , ret_len  # ( batch_size ,<conditionid> , max_len,max_tok_num )






    def forward(self,q,q_len,col_inp_var, name_length,col_length,gt_cond=None,gt_where=None):



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
        #cond_num_score = self.softmax(cond_num_score)


        #-------------------------------------------------------------------------------------

        

        # Portion for  predicting the columns to be included in the WHERE clause

        #---------------------------------------------------------------------------------------


        e_cond_col , _  = column_encode( self.cond_col_name_enc, col_inp_var,name_length,col_length  )
        h_col_enc ,  _  = run_lstm(self.cond_col_lstm,q,q_len)

        col_att_val     = torch.bmm( e_cond_col , self.cond_col_att(h_col_enc).transpose(1,2) )

        for i, num in enumerate( q_len ):

            if num<max_x_len:
                col_att_val[ i , : , num: ] = -100


        col_att = self.softmax( col_att_val.view((-1,max_x_len))).view(batch_size, -1, max_x_len)

        k_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3) ).sum(2)


        cond_col_score = self.cond_col_out( self.cond_col_out_k(k_cond_col) + self.cond_col_out_col(e_cond_col )).squeeze()

        max_col_num = max(col_length)

        for i , num in enumerate(col_length):
            if num<max_col_num:
                cond_col_score[i,num:] = -100

        cond_col_score = self.softmax(cond_col_score)
        

        #------------------------------------------------------------------------

        # Portion for prediciting the operators ( GREATER THAN ,LESS THAN ,EQUAL TO ) to be used
        # against each column predicted by the above portion

        #------------------------------------------------------------------------


        
        chosen_col_gt = [] 

        if gt_cond is None:


            cond_nums = np.argmax( cond_num_score.data.cpu().numpy() , axis=1  )    # Get the number of conditions corresponding to each question

            col_scores = cond_col_score.data.cpu().numpy()
            chosen_col_gt = [  list(np.argsort(-col_scores[b])  [ : cond_nums[b]]) for b in range(len(cond_nums)) ] 

        else : 

            chosen_col_gt = gt_cond



        # chosen col_gt contains the indexes of column as a list as each element of chosen_col_gt


        e_cond_col , _  = column_encode( self.cond_op_name_enc, col_inp_var,name_length,col_length )

        col_emb = []

        for i in range(batch_size):
            
            cur_col_emb = torch.stack( [e_cond_col[i,x] for x in chosen_col_gt[i]]
                    + [e_cond_col[i,0]]*( 4 - len(chosen_col_gt[i])
                ))
            # 4 is chosen as the the maximum number of condtions restricted is 4

            col_emb.append(cur_col_emb)

        col_emb = torch.stack(col_emb)  # Convert the array  to  a torch tensor by stacking along the elements
        
        h_op_enc , _  = run_lstm(self.cond_op_lstm , q ,q_len)

        # Column attention
        
        
        op_att_val = torch.matmul( self.cond_op_att(h_op_enc).unsqueeze(1) , col_emb.unsqueeze(3) ).squeeze()
        
        for  i ,num in enumerate(q_len):
            if num<max_x_len : 
                op_att_val [ i ,: , num:] = -100
        op_att = self.softmax( op_att_val.view(batch_size*4,-1) ).view(batch_size, 4,-1 ) 
        k_cond_op = ( h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)


        cond_op_score = self.cond_op_out( self.cond_op_out_k(k_cond_op) + self.cond_op_out_col(col_emb) ).squeeze()

        
        #-------------------------------------------------------------------------

        # Portion for predicting the values corresponding to each condition using
        # a Pointer Network

        #------------------------------------------------------------------------

        h_str_enc , _ = run_lstm(self.cond_str_lstm, q ,q_len)
        e_cond_col, _ = column_encode( self.cond_str_name_enc, col_inp_var,name_length,col_length  )

        col_emb = [] 
        for b in range(batch_size):

            cur_col_emb = torch.stack( 
                        [ e_cond_col[b,x] for x in chosen_col_gt[b] ] +
                        [ e_cond_col[b,0]]*( 4- len(chosen_col_gt[b]) )
                        
                    )
            col_emb.append(cur_col_emb)

        col_emb = torch.stack(col_emb)

        

        ####### Ground truth condtions
        
        if gt_where is not None:
            
            gt_tok_seq , gt_tok_len = self.gen_gt_batch(gt_where)
            g_str_s_flat ,_  =  self.cond_str_decoder(
                        gt_tok_seq.view(batch_size*4,-1,self.max_tok_num)
                    )

            g_str_s = g_str_s_flat.contiguous().view(batch_size,4,-1,self.hidden_dim  )   #Create new tensor

            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            g_ext = g_str_s.unsqueeze(3)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)


            cond_str_score = self.cond_str_out(
                    
                        self.cond_str_out_g(g_ext) + 
                        self.cond_str_out_h(h_ext) +
                        self.cond_str_out_col(col_ext)
                        
                    ).squeeze()
        
            for i , num in enumerate(q_len ):

                if num < max_x_len:
                    cond_str_score[b,:,:,num:] = -100

        else:


            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)

            scores = []

            t = 0 

            init_inp = np.zeros( (batch_size *4 , 1, self.max_tok_num) , dtype=np.float32 )
            init_inp[:,0,0] = 1

            ## CUDA Here below
            cur_inp  = Variable(torch.from_numpy(init_inp))

            cur_h = None

            while t < 50:

                if cur_h:
                    g_str_s_flat , cur_h = self.cond_str_decoder(cur_inp,cur_h)
                else:
                    g_str_s_flat , cur_h = self.cond_str_decoder(cur_inp)

                g_str_s = g_str_s_flat.view(batch_size,4,1,self.hidden_dim)
                g_ext = g_str_s.unsqueeze(3)


                # Compute the score

                cur_cond_str_score = self.cond_str_out (
            
                        self.cond_str_out_h(h_ext) + 
                        self.cond_str_out_g(g_ext) +
                        self.cond_str_out_col(col_ext)

                ).squeeze()


                for i ,num in enumerate(q_len):
                    if num < max_x_len:
                        cur_cond_str_score[ b , : ,num:] = -100

                scores.append(cur_cond_str_score)



                _ , ans_tok_var = cur_cond_str_score.view( batch_size*4,max_x_len).max(1)
                ans_tok = ans_tok_var.data.cpu()

                data = torch.zeros(batch_size*4 , self.max_tok_num).scatter_(
                            1, ans_tok.unsqueeze(1) , 1
                 )   


                # CUDA below
                cur_inp = Variable(data)
                cur_inp = cur_inp.unsqueeze(1)
                t+=1


            cond_str_score = torch.stack(scores,dim=2)

            for i , num in enumerate(q_len):
                if num < max_x_len:
                    cond_str_score[b,:,:,num:] = -100



        

        return cond_num_score,cond_col_score,cond_op_score,cond_str_score

       
