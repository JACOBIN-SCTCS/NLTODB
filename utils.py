import json 
import torch
from torch.autograd import Variable
import torch.nn as nn
from preprocessing import *
from torch.utils.data import Dataset,DataLoader



# Using a custom dataset class for loading the data
class SQLDataset(Dataset):


    #   Params
    #   file_path :  type of the file  choices are train/test/dev
    
    def __init__(self,file_path):

        self.sql_data , self.table_data =  load_dataset(file_path)


    # Function that needs to be overloaded when inheriting from Dataset Class
    def __len__(self):
        return len(self.sql_data)


       
    # Function that needs to be overloaded when inheriting from Dataset Class
    def __getitem__(self,idx):
        
        sql_item = self.sql_data[idx]
        sql_item_sql = sql_item['sql']

        table_id = sql_item['table_id']
        question_tokens = unicodeToAscii(sql_item['question']).split(' ')
        sql_query =  sql_item_sql["agg"]

        return {
            'table_id': table_id,
            'question_tokens' : question_tokens,
            'sql_query': sql_query
        }

    

# For testing purposes only Uncomment the code for testing
'''

sq = SQLDataset('train')
sql_dataloader = DataLoader(sq,batch_size=64,num_workers=3)
g=next(iter(sql_dataloader))
print(g['question_tokens'])
'''


