import json 
import torch
from torch.autograd import Variable
import torch.nn as nn
from preprocessing import load_dataset,unicodeToAscii,splitColumnNames
from torch.utils.data import Dataset,DataLoader


def collate_fn(batch):
    # "Puts each data field into a tensor with outer dimension batch size"
    return {key: [d[key] for d in batch] for key in batch[0]}



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
        column_headers = splitColumnNames( self.table_data[table_id]['header'] )
        column_num = len(self.table_data[table_id]['header'])
        sql_query =  sql_item_sql["agg"]

        return {
            'table_id': table_id,
            'question_tokens' : question_tokens,
            'column_headers' : column_headers,
            'column_num'    : column_num,
            'sql_query': sql_query
        }

    





def train_model( model, n_epochs , optimizer,dataloader ):

    model.train()

    for e in range(n_epochs):

        epoch_loss = 0

        for data in dataloader:

            model.zero_grad()
            optimizer.zero_grad()

            scores = model(data['question_tokens'] , data['column_headers'] ,(True,None,None))
            loss = model.loss(scores,data['sql_query'])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5 )
            optimizer.step()

        
            epoch_loss += loss.item()
        
        
        print('Loss {} ----- {}'.format(e,  epoch_loss / len(dataloader) ))

    torch.save(model.state_dict(), 'saved_models/agg_model.pth')
    print(model.state_dict())

        



# For testing purposes only Uncomment the code for testing


'''sq = SQLDataset('train')
sql_dataloader = DataLoader(sq,batch_size=5,num_workers=1,collate_fn=collate_fn)
g=next(iter(sql_dataloader))
print(g['column_headers'])
'''






