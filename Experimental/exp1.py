

import numpy as np
import torch
import torch.nn as nn


def bert_question_encode(q_arr , bert_model,bert_tokenizer):

    batch_size  = len(q_arr)
    

    questions = [] 
    for i in q_arr:

        questions.append(' '.join(i ))
    
    q_len = np.zeros((batch_size),dtype=np.int_)

    q = []
    for question in questions:

        tokenized_questions = bert_tokenizer.encode(question,add_special_tokens=True)
        q.append(tokenized_questions)


    for i in range(len(q)):
        q_len[i] = len(q[i])

    max_question_length = int(max(q_len))

    batch_q_embedding = np.zeros((batch_size,max_question_length),dtype=np.int_)

    for i in range(batch_size):

        batch_q_embedding[i,:q_len[i]] = q[i]
    
    questions_embedding = torch.tensor(batch_q_embedding)
    
    bert_model.eval()
    outputs = None

    with  torch.no_grad():
        outputs = bert_model(questions_embedding)

    return outputs[0],q_len




def bert_column_encode( cols_array, bert_model, bert_tokenizer,bert_hidden_size=256):

    columns = [] 
    batch_size = len(cols_array)
    
    for b in range(batch_size):
        current_ques = [] 
        for col in cols_array[b]:
            current_ques.append(' '.join(col))
        columns.append(current_ques)


    col_len = np.zeros((batch_size),dtype=np.int_)
    for i in range(batch_size):
        col_len[i] = len(columns[i])


    column_embeddings = [] 

    with torch.no_grad():

        for b in range(batch_size):
            current_ques = []
            for c in columns[b]:
                encoded_name = torch.tensor([ bert_tokenizer.encode(c,add_special_tokens=True) ])
                encoded = bert_model(encoded_name)

                current_ques.append(encoded[1][0])
            column_embeddings.append(current_ques)
    

    max_col_length = int(max(col_len))

    column = torch.zeros((batch_size,max_col_length,bert_hidden_size))

    for b in range(batch_size):
        for l in range(int(col_len[b])):
            column[b,l,:] = column_embeddings[b][l]


    return column , col_len


def sqlova_gt_where(questions,conditions,bert_tokenizer):
    
    batch_size = len(conditions)
    gt_conds = []
    
    question= []
    
    for cur_q in questions:
        current = ' '.join(cur_q)
        current_special = '[CLS] ' + current + ' [SEP]'
        question.append(current_special)
    
    
    for b in range(batch_size):
        con = []
        for cond in conditions[b]:
            con.append(cond[2])
            
        gt_conds.append(con)
    
    tokenized_questions = []
    
    for q in question:
        tokenized_questions.append(bert_tokenizer.tokenize(q))
        
        
    tokenized_conditions = []
    for b in range(batch_size):
        current_cond = []
        for i  in gt_conds[b]:
            current_cond.append(bert_tokenizer.tokenize(str(i)))
        tokenized_conditions.append(current_cond)
        
    gt_cond_str = []
    
    for b in range(batch_size):
        conds= []
        for i in tokenized_conditions[b]:
            flag= False
            for j in range(len(tokenized_questions[b])):
                if tokenized_questions[b][j:j+len(i)] ==i:
                    flag = True
                    conds.append([j,j+len(i)-1])
                    break
                
            if flag==False:
                try:
                    
                    start_idx = tokenized_questions[b].index(tokenized_conditions[b][0])
                except ValueError:
                    start_idx = len(tokenized_questions[b])-1
                    
                for j in range(1,len(tokenized_conditions[b])):
                    if start_idx+j == len(tokenized_questions[b]):
                        break
                    if tokenized_conditions[b][j] != tokenized_questions[b][start_idx+j]:
                        break
                conds.append([start_idx,(start_idx+j-1)])
                
                
        gt_cond_str.append(conds)

    return gt_cond_str




def sqlova_gen_query_acc( cond_scores, questions ):

    cond_num_score, cond_col_score, cond_op_score, cond_str_score = [
                x.data.cpu().numpy()  for x in cond_scores
            ]

    pred_cond = [] 

    for b in range(len(cond_num_score)):

        b_cond = []
        cond_num = np.argmax(cond_num_score[b])

        #all_toks = ['<BEG>'] + questions[b] + ['<END>']
        max_idxes = np.argsort(-cond_col_score[b])[:cond_num]

        for i in range(cond_num):
            cur_cond = [] 
            cur_cond.append(max_idxes[i])
            cur_cond.append(np.argmax(cond_op_score[b][i]))
            cur_cond_str_toks = []

            start_idx, end_idx = np.argmax(cond_str_score[b][i],axis=0)
            start_idx = int(start_idx)
            end_idx   = int(end_idx)
                
            if start_idx > end_idx:
                temp = start_idx
                start_idx = end_idx
                end_idx = temp
                
            cur_cond_str_toks =  questions[b][start_idx:(end_idx+1)]
                
                
                
            '''
                
                str_tok = np.argmax(str_score[:len(all_toks)])
                str_val = all_toks[str_tok]
                if str_val == '<END>':
                    break
                cur_cond_str_toks.append(str_val)
                
            '''


            modif_list = []
            for j in cur_cond_str_toks:
                if j not in modif_list and j!='[CLS]' and j!='[SEP]':
                    modif_list.append(j)

            cur_cond_str_toks = modif_list
            #cur_cond.append(merge_tokens( cur_cond_str_toks, "".join(questions[b])  ))
            #cur_cond.append(' '.join(cur_cond_str_toks))
            
            text = ' '.join([x for x in cur_cond_str_toks])
            fine_text = text.replace(' ##', '')
            cur_cond.append(fine_text)
            
            
            b_cond.append(cur_cond)
        pred_cond.append(b_cond)

    return pred_cond

                
            

        
    
    





