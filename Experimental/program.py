

from exp1 import sqlova_gen_query_acc , sqlova_gt_where , bert_question_encode
from exp2 import SqlovaCondPredictor
from transformers import BertModel,BertTokenizer
import torch
import numpy as np


model = BertModel.from_pretrained('google/bert_uncased_L-8_H-256_A-4')
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-8_H-256_A-4')

questions = [['what', 'is', 'the', 'estimated', 'deaths', 'with', 'operational', 'period', 'of', '17', 'march', '1942', '–', 'end', 'of', 'june', '1943'], ['what', 'is', 'the', 'occupied', 'territory', 'with', 'operational', 'period', 'of', 'may', '1940', '–', 'january', '1945'], ['how', 'many', 'poles', 'had', 'an', 'average', 'finish', 'of', '19.1'], ['who', 'had', 'the', 'fastest', 'lap', 'at', 'the', 'brazilian', 'grand', 'prix'], ['what', 'are', 'the', 'original', 'air', 'dates', 'with', 'a', 'production', 'code', 'of', '2394087'], ['who', 'took', 'test', '#4'], ['what', 'is', 'the', 'fastest', 'lap', 'with', 'pole', 'position', 'of', 'gilles', 'villeneuve'], ['how', 'high', 'is', 'the', 'chance', 'that', 'player', '1', 'wins', 'if', 'player', '2', 'has', 'an', '88.29%', 'chance', 'of', 'winning', 'with', 'the', 'choice', 'of', 'r', 'rb'], ['when', 'did', 'the', 'player', 'from', 'hawaii', 'play', 'for', 'toronto'], ['what', 'is', 'the', 'name', 'of', 'the', 'green', 'house']]

gt_cond = [[[2, 0, '17 March 1942 – end of June 1943']], [[2, 0, 'May 1940 – January 1945']], [[7, 0, '19.1']], [[1, 0, 'Brazilian Grand Prix']], [[6, 0, '2394087']], [[0, 0, 4]], [[4, 0, 'Gilles Villeneuve']], [[3, 0, '88.29%'], [1, 0, 'R RB']], [[5, 0, 'Hawaii']], [[4, 0, 'Green']]]




q, q_len = bert_question_encode(questions,model,tokenizer)
 
col_len = np.array([6, 6, 11, 7, 7, 6, 9, 5, 6, 5])
col = torch.randn(10,11,256)



cond = SqlovaCondPredictor(256,100)

scores = cond(q,q_len,col,col_len)

print(torch.argmax(scores[0],dim=1))


tokenized_questions = []
for b in range(len(q)):
    
    q = ' '.join(questions[b])
    aug = '[CLS] ' +q + ' [SEP]'
    tokenized_questions.append(tokenizer.tokenize(aug))

generated_query = sqlova_gen_query_acc(scores,tokenized_questions)

print(generated_query)
print('\n\n')
print(gt_cond)



