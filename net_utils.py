import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable



'''
    Function name : run_lstm
    
    Parameters:
    __________________

        lstm : A pytorch lstm object
        inp  : Torch tensor of shape [batch_size, max_length, embedding dimension ]
                max_length is the length of largest question in the batch

        inp_length : Array containing length of each question in batch has size batch_size


''' 

def run_lstm(lstm, inp , inp_length,hidden=None ):

    sort_perm = np.argsort(inp_length)[::-1].copy()   # a numpy array containing indices of inp in order of their correspondng length descending
    sort_inp_len = inp_length[sort_perm]              # input length in decreasing order
    sort_perm_inv = np.argsort(sort_perm)             # return inp to order in which received.


    lstm_inp = nn.utils.rnn.pack_padded_sequence( inp[sort_perm] , sort_inp_len,batch_first=True )

    if hidden is None:
      hidden=None
    else:
        hidden = (hidden[0][:,sort_perm] , hidden[1][:,sort_perm])      # Permute the hidden state corresponding to sort_perm

    output , ret_h  = lstm(lstm_inp,hidden)
    
    ret_output = nn.utils.rnn.pad_packed_sequence(output,batch_first=True)[0][sort_perm_inv]

    ret_hidden = ( ret_h[0][:, sort_perm_inv] , ret_h[0][:, sort_perm_inv])
    return ret_output,ret_hidden



def column_encode( lstm ,  col_inp_var, name_length ,col_length ):

    col_hidden , col_out =  run_lstm( lstm, col_inp_var, name_length )
    
    name_out = col_hidden[ tuple(range(len(name_length))) , name_length-1  ]

    ret = torch.FloatTensor( len(col_length) , max(col_length) , name_out.size()[1]  ).zero_()

    st = 0 
    for  i ,cur_len in enumerate(col_length):
        ret[i, :cur_len] = name_out.data[st: st+cur_len]
        st += cur_len

    ret_var = Variable(ret)

    return ret_var,col_length




