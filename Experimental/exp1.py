

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





