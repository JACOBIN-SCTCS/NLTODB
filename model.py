import torch
import torch.nn as nn
from agg_predictor import AggPredictor
from wordembedding import WordEmbedding
from torch.autograd import Variable
import numpy as np



class Model(nn.Module):

    def __init__(self , hidden_dim , embed_dim , word_emb):
        
        super().__init__()
        


        self.agg_ops =  ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim


        self.agg_predictor = AggPredictor( embed_dim , hidden_dim ) 

        self.ce = nn.CrossEntropyLoss()

        self.wordembedding = word_emb


    def forward(self, question , columns , pred_entry ):

        pred_agg , pred_sel, pred_cond = pred_entry

        embedding , length = self.wordembedding.gen_x_batch(question,columns)

        agg_score = None
        
        batch_size = len(question)
        
        if pred_agg:
            hidden = ( torch.zeros(2,batch_size,int(self.hidden_dim/2)) , torch.zeros(2,batch_size,int(self.hidden_dim/2))  )
            agg_score = self.agg_predictor.forward(embedding,length, hidden)


        return agg_score

        
   
    def  loss(self, score, truth ):
        loss = 0 
        agg_truth = torch.from_numpy(np.asarray(truth))
        agg_truth_var = Variable(agg_truth)
        
        loss +=  self.ce(score,agg_truth_var)

        return loss


       
