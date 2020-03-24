import torch
import torch.nn as nn
from agg_predictor import AggPredictor
from cond_predictor import CondPredictor
from select_predictor import SelectionPredictor
from wordembedding import WordEmbedding
from torch.autograd import Variable
import numpy as np



class Model(nn.Module):

    def __init__(self , hidden_dim , embed_dim , word_emb):
        
        super().__init__()
        


        self.agg_ops =  ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        

        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.num_layers = 2
        self.max_tok_num = 200
        

        self.agg_predictor = AggPredictor( embed_dim , hidden_dim , self.num_layers) 
        self.cond_predictor = CondPredictor( embed_dim , hidden_dim , self.num_layers, self.max_tok_num )
        self.sel_predictor  = SelectionPredictor(embed_dim,hidden_dim , self.num_layers)

        self.ce = nn.CrossEntropyLoss()

        self.wordembedding = word_emb

        self.sigmoid  = nn.Sigmoid()


    def forward(self, question , columns ,  pred_entry  , where_col=None, gt_where=None):

        pred_agg , pred_sel, pred_cond = pred_entry

        embedding , length = self.wordembedding.gen_x_batch(question,columns)
        col_inp_var , name_length , col_length = self.wordembedding.gen_column_batch(columns)


        agg_score = None
        sel_score = None
        cond_score = None
        
        batch_size = len(question)
        
        if pred_agg:
            hidden = ( torch.zeros(self.num_layers*2,batch_size,int(self.hidden_dim/2)) , torch.zeros(self.num_layers*2,batch_size,int(self.hidden_dim/2))  )
            agg_score = self.agg_predictor.forward(embedding,length, hidden)


        
        if pred_sel:
            hidden = (torch.zeros(self.num_layers*2,batch_size,int(self.hidden_dim/2)) ,  torch.zeros(self.num_layers*2,batch_size,int(self.hidden_dim/2))  )
            sel_score =self.sel_predictor.forward(embedding,length,col_inp_var,name_length,col_length,hidden)



        if pred_cond:
            cond_score = self.cond_predictor.forward( embedding , length, col_inp_var,name_length,col_length,where_col,gt_where )


        
        return ( agg_score , sel_score, cond_score )

        
   
    def  loss(self, score, truth , pred_entry ):
        
        pred_agg , pred_sel,pred_cond = pred_entry
        
        loss = 0 
        if pred_agg:
            
            agg_truth = torch.from_numpy(np.asarray(truth[0]))
            agg_truth_var = Variable(agg_truth)
        
            loss +=  self.ce(score[0],agg_truth_var)

        if pred_sel:
            
            sel_truth = torch.from_numpy(np.asarray(truth[1]))
            sel_truth_var = Variable(sel_truth)

            loss+=   self.ce(score[1],sel_truth_var)
        
        if pred_cond:



            # Loss for the number of conditions
            cond_num_score , cond_col_score , cond_op_score,cond_str_score = score[2]


            cond_col_num = torch.from_numpy(np.asarray(truth[2]))
            cond_col_num_var = Variable(cond_col_num)

            loss += self.ce(cond_num_score, cond_col_num_var )
        



            # Loss for columns associated with each condition
            T = len( cond_col_score[0] )  #Maximum number of tokens
            batch_size = len(cond_col_score)
            truth_prob = np.zeros(  ( batch_size ,T) , dtype=np.float32)
            for b in range(batch_size):
                if len( truth[3][b]  ) > 0:

                    truth_prob[b][ list( truth[3][b]) ] =1

            #CUDA cond_col_truth_var
            cond_col_truth_var = Variable(torch.from_numpy(truth_prob))
        
            cond_col_prob = self.sigmoid( cond_col_score )
            bce_loss = -torch.mean( 
                    
                    3*( cond_col_truth_var *torch.log(cond_col_prob+1e-10)) +
                    (1-cond_col_truth_var)*torch.log(1-cond_col_prob+1e-10)

                    )

            loss += bce_loss
            

           # LOSS for operators corresponding to each condition
            for b in range(batch_size):
                if len( truth[4][b] ) ==0:
                    continue             #Continue if there is no condition

                #CUDA cond_op_truth
                cond_op_truth_var = Variable( torch.from_numpy( np.array( truth[4][b]  ) ) )
                cond_op_pred  = cond_op_score[ b , :len(truth[4][b] )]
                loss += (  self.ce(cond_op_pred,cond_op_truth_var) / batch_size  )

            
            # LOSS for strings in each condition Crossentropy Loss against one hot
            # encoding of names
        
            for b in range(len( truth[5]  )):
                for idx in range(len(truth[5][b])):
                    cond_str_truth = truth[5][b][idx]
                    if len(cond_str_truth) == 1:
                        continue
                    data = torch.from_numpy(np.array(cond_str_truth[1:]) )

                    # CUDA cond_str_truth_var
                    cond_str_truth_var = Variable(data)
                    str_end = len(cond_str_truth)-1

                    cond_str_pred = cond_str_score[b,idx, :str_end]
                    loss += ( self.ce(cond_str_pred,cond_str_truth_var)/ ( len(truth[5])* len(truth[5][b]) ) )

    

        return loss

    
    def validation_loss(self,score,truth , pred_entry ):
        
        pred_agg , pred_sel , pred_cond = pred_entry
        
        agg_loss = 0.0
        sel_loss = 0.0
        cond_loss= 0.0

        if pred_agg :
            agg_truth = torch.from_numpy( np.asarray( truth[0] ) )
            agg_truth_var = Variable(agg_truth)

            agg_loss = self.ce( score[0] , agg_truth_var )


        if pred_sel:
            sel_truth = torch.from_numpy( np.asarray( truth[1] ) )
            sel_truth_var = Variable(sel_truth)
            sel_loss = self.ce( score[1] , sel_truth_var )
            
            
        if pred_cond:



            # Loss for the number of conditions
            cond_num_score , cond_col_score , cond_op_score,cond_str_score = score[2]


            cond_col_num = torch.from_numpy(np.asarray(truth[2]))
            cond_col_num_var = Variable(cond_col_num)

            cond_loss += self.ce(cond_num_score, cond_col_num_var )
        



            # Loss for columns associated with each condition
            T = len( cond_col_score[0] )  #Maximum number of tokens
            batch_size = len(cond_col_score)
            truth_prob = np.zeros(  ( batch_size ,T) , dtype=np.float32)
            for b in range(batch_size):
                if len( truth[3][b]  ) > 0:

                    truth_prob[b][ list( truth[3][b]) ] =1

            #CUDA cond_col_truth_var
            cond_col_truth_var = Variable(torch.from_numpy(truth_prob))
        
            cond_col_prob = self.sigmoid( cond_col_score )
            bce_loss = -torch.mean( 
                    
                    3*( cond_col_truth_var *torch.log(cond_col_prob+1e-10)) +
                    (1-cond_col_truth_var)*torch.log(1-cond_col_prob+1e-10)

                    )

            cond_loss += bce_loss
            

           # LOSS for operators corresponding to each condition
            for b in range(batch_size):
                if len( truth[4][b] ) ==0:
                    continue             #Continue if there is no condition

                #CUDA cond_op_truth
                cond_op_truth_var = Variable( torch.from_numpy( np.array( truth[4][b]  ) ) )
                cond_op_pred  = cond_op_score[ b , :len(truth[4][b] )]
                cond_loss += (  self.ce(cond_op_pred,cond_op_truth_var) / batch_size  )

            
            # LOSS for strings in each condition Crossentropy Loss against one hot
            # encoding of names
        
            for b in range(len( truth[5]  )):
                for idx in range(len(truth[5][b])):
                    cond_str_truth = truth[5][b][idx]
                    if len(cond_str_truth) == 1:
                        continue
                    data = torch.from_numpy(np.array(cond_str_truth[1:]) )

                    # CUDA cond_str_truth_var
                    cond_str_truth_var = Variable(data)
                    str_end = len(cond_str_truth)-1

                    cond_str_pred = cond_str_score[b,idx, :str_end]
                    cond_loss += ( self.ce(cond_str_pred,cond_str_truth_var)/ ( len(truth[5])* len(truth[5][b]) ) )

    

    



        return (agg_loss,sel_loss,cond_loss)


       
