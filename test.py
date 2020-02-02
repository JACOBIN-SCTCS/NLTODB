from utils import SQLDataset,collate_fn
from wordembedding import WordEmbedding
from extract_vocab import load_word_emb
from torch.utils.data import Dataset,DataLoader



N_word = 50 

word_embed = load_word_emb('glove/glove.6B.50d.txt')

sq = SQLDataset('train')
sql_dataloader = DataLoader(sq,batch_size=5,num_workers=1,collate_fn=collate_fn)
g=next(iter(sql_dataloader))

word_emb = WordEmbedding(N_word,word_embed)
embeddings,len = word_emb.gen_x_batch(g['question_tokens'],g['column_headers'])
print(embeddings.shape)



