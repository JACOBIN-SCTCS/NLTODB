from preprocessing import load_dataset,unicodeToAscii
import torch
import numpy as np
import json


#N_WORD = 50


# Loads the dataset for use for creating the vocabulary out of question words and 
# column names
#sql_data,table_data = load_dataset('train')



'''
word_to_idx = { 'UNK':0,'<BEG>':1,'<END>':2 }                               # Dictionary storing index of each word which is used as an index into an array containing embeddings 
word_num = 3                                                                # Counter which stores the number of words in the vocabulary
embs = [ np.zeros(N_WORD,dtype=np.float32) for _ in range(word_num) ]       # A numpy array which stores the embeddings of words so far . It is shape of a 2D matrix row correspond to each word
'''



def load_word_emb(filename,load_used=False):
    if not load_used:

        print('Loading word embeddings from file')
        ret = {} 
        with open(filename,'r') as f:
            for id,line in enumerate(f):
                if(id>= 10000):
                    break
                
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0].lower()] = np.array([ float(x) for x in info[1:]] )
        return ret

    else: 

        print ('Loading preformatted embeddings')
        with open('word2idx.json') as f:
            w2i = json.load(f)
        with open('usedwordemb.npy') as f:
            word_emb_val = np.load(f)
        return w2i,word_emb_val






'''

word_emb = load_word_emb('glove/glove.6B.50d.txt')


def check_and_add(tok):
    global word_num

    if tok not in word_to_idx and tok in word_emb:
        word_to_idx[tok] = word_num
        word_num = word_num + 1
        embs.append(word_emb[tok])


for sql in sql_data:
    for tok in unicodeToAscii(sql['question']).split(' '):
        check_and_add(tok)



emb_array = np.stack(embs,axis =0 )
with open('word2idx.json','w') as outf:
    json.dump(word_to_idx,outf)

np.save( open('usedwordemb.npy','wb') , emb_array )

'''



