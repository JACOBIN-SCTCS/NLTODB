import json
from utils import gen_query_acc
from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from torch.utils.data import Dataset,DataLoader
from model import Model
from extract_vocab import load_word_emb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim



def gen_query_acc_for_file( cond_scores, questions ):

    cond_num_score, cond_col_score, cond_op_score, cond_str_score = [
                x.data.cpu().numpy()  for x in cond_scores
            ]

    pred_cond = [] 

    for b in range(len(cond_num_score)):

        b_cond = []
        cond_num = np.argmax(cond_num_score[b])

        all_toks = ['<BEG>'] + questions[b] + ['<END>']
        max_idxes = np.argsort(-cond_col_score[b])[:cond_num]

        for i in range(int(cond_num)):
            cur_cond = [] 
            cur_cond.append(int(max_idxes[i]))
            cur_cond.append(int(np.argmax(cond_op_score[b][i])))
            cur_cond_str_toks = []

            for str_score in cond_str_score[b][i]:
                str_tok = np.argmax(str_score[:len(all_toks)])
                str_val = all_toks[str_tok]
                if str_val == '<END>':
                    break
                cur_cond_str_toks.append(str_val)

            # Modif Codes can be changed
            modif_list = []
            for j in cur_cond_str_toks:
                if j not in modif_list and j!='<BEG>':
                    modif_list.append(j)

            cur_cond_str_toks = modif_list
            #cur_cond.append(merge_tokens( cur_cond_str_toks, "".join(questions[b])  ))
            cur_cond.append(' '.join(cur_cond_str_toks))
            b_cond.append(cur_cond)
        pred_cond.append(b_cond)

    return pred_cond





def get_output_as_file(model,test_loader):

    print("\n Creating file\n")
    model.agg_predictor.load_state_dict( torch.load('saved_models/agg_predictor.pth'))    
    model.sel_predictor.load_state_dict( torch.load('saved_models/sel_predictor.pth'))
    model.cond_predictor.load_state_dict(torch.load('saved_models/cond_predictor.pth'))

    model.eval()

    out_file = open('pred_out.jsonl','w')

    for data in test_loader:

        scores  = model(data['question_tokens'] , data['column_headers'],(True,True,True),
                  data['where_col'], data['gt_where']
                )
        
        out_agg = torch.argmax( torch.exp(scores[0]),dim=1)
        out_sel = torch.argmax( torch.exp(scores[1]),dim=1)
        pred_cond = gen_query_acc_for_file( scores[2], data['question_tokens']  )
        
        for b in range(len(out_agg)):

            current_data = {}
            current_data["query"] = {}

            current_data["query"]["sel"] = int(out_sel[b])
            current_data["query"]["agg"] = int(out_agg[b])
            current_data["query"]["conds"] = pred_cond[b]

            json.dump(current_data,out_file)
            out_file.write("\n")

    out_file.close()


filename = 'glove/glove.42B.300d.txt'
N_word = 300
batch_size = 64
hidden_dim = 100
n_epochs = 5


word_embed = load_word_emb(filename)
test = SQLDataset('test')
test_loader = DataLoader(test,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

word_emb = WordEmbedding(N_word,word_embed)
model = Model(hidden_dim,N_word,word_emb)


get_output_as_file(model,test_loader)
print("\n Created file")



