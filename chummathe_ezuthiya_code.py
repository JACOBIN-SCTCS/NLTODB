from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from extract_vocab import load_word_emb
from torch.utils.data import Dataset,DataLoader
from net_utils import run_lstm
from model import Model
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from utils import train_model,test_model
from model import Model
from torch.autograd import Variable

N_word = 50 
batch_size =10
hidden_dim = 100

word_embed = load_word_emb('glove/glove.6B.50d.txt')



train =  SQLDataset('train')
#train , valid   = SQLDataset('train') , SQLDataset('dev')
train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)

#valid_dataloader = DataLoader(valid,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=collate_fn)



#test = SQLDataset('test')
#test_dataloader = DataLoader(test,batch_size = batch_size, shuffle=True, num_workers=1,collate_fn=collate_fn)


g=next(iter(train_dataloader))
#print(g)



word_emb = WordEmbedding(N_word,word_embed)

mod = Model(hidden_dim,N_word,word_emb)
train_entry =(None,None,True)

agg , sel , cond = mod(g['question_tokens'] , g['column_headers'], train_entry , g['where_col'], g['gt_where'] )

#print(cond[0].shape)
#print(cond[1].shape)
#print(cond[2].shape)
#print(cond[3].shape)

loss = mod.validation_loss((agg,sel,cond), (g['agg'] , None,g['cond_num'],g['where_col'],g['where_op'],g['gt_where']) , train_entry )
print(loss[2])

'''
loss = 0 

criterion = nn.CrossEntropyLoss()
gt_where = g['gt_where']


for b in range(len(gt_where)):
    for idx in range(len(gt_where[b])):
        cond_str_truth = gt_where[b][idx]
        if len(cond_str_truth) == 1:
            continue
        data = torch.from_numpy(np.array(cond_str_truth[1:]))
                   
        cond_str_truth_var = Variable(data)
        str_end = len(cond_str_truth)-1
        cond_str_pred = cond[3][b, idx, :str_end]
        loss += (criterion(cond_str_pred, cond_str_truth_var) \
                            / (len(gt_where) * len(gt_where[b])))


#print(gt_where)
#print(gt_where[0])

#print(cond[3].shape)

print(loss) 

'''


'''
loss =0
where_op = g['where_op']

for b in range(batch_size):
    if len(where_op[b]) ==0:
        continue

    cond_op_truth_var = Variable(torch.from_numpy(np.array(where_op[b])))
    cond_op_pred = cond[2][b , :len(where_op[b]) ]
    loss+= ( ( nn.CrossEntropyLoss()(cond_op_pred,cond_op_truth_var) ) / batch_size  )

print(loss)
'''


#where_col = g['cond_num']
#print(where_col)
#where_col = Variable(torch.from_numpy(np.array(where_col)))
#loss = nn.CrossEntropyLoss()(cond[0],where_col)
#print(loss)

'''
gt_col = g['where_col']

T  = len( cond[1][0] )
truth_prob = np.zeros((batch_size,T),dtype=np.float32)
for b in range(batch_size):
    if len(gt_col[b]) >0:
        truth_prob[b][ list( gt_col[b] ) ]  =1 


cond_col_truth_var = Variable(torch.from_numpy(truth_prob))
cond_col_prob = nn.Sigmoid()(cond[1])
bce_loss = -torch.mean(
            3*(cond_col_truth_var * torch.log(cond_col_prob+1e-10) ) +
            (1 - cond_col_truth_var)* torch.log( 1- cond_col_prob+1e-10  )
        
        )
print(bce_loss)



weight_tensor = torch.tensor([ float(3.0) for _ in range(T)])
alter_bc_loss = nn.BCELoss(weight_tensor)(cond_col_prob,cond_col_truth_var)
print(alter_bc_loss)
'''



#embeddings,length = word_emb.gen_x_batch(g['question_tokens'],g['column_headers'])
#print(embeddings)
#sql_query = np.asarray(g['sql_query'])
#agg_tensor = torch.from_numpy(sql_query)
#print(agg_tensor.shape)

#name_inp_var , name_len , col_len = word_emb.gen_column_batch( g['column_headers'])

#print(g['column_headers'] )
#print(name_inp_var.shape)
#print(name_len)
#print(col_len)


#rnn = nn.LSTM(N_word,hidden_dim,batch_first=True)
#ret_var,col_len = column_encode( rnn , name_inp_var,name_len,col_len )
#print(ret_var)



#model = Model(hidden_dim,N_word,word_emb)



#agg_model = AggPredictor(N_word,hidden_dim)
#epochs = 5

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=0.01)

#train_model(model,epochs,optimizer,train_dataloader,valid_dataloader)




#testmodel = Model(hidden_dim,N_word,word_emb)

#testmodel.load_state_dict( torch.load('saved_models/agg_model.pth') )


#test_model( testmodel,test_dataloader)




#question = [ 'What is the total salary of employee 3'.split(' ') ,  'What is the total salary of employee 3'.split(' ')  ] 
#columns =[ [ ['id'],['batch'],['name']]  ,   [ ['id'],['batch'],['name']]  ]
#scores = test_model( question, columns , (True,None,None) )

#print( torch.exp(scores))


