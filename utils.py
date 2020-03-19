import json 
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from preprocessing import load_dataset,unicodeToAscii,splitColumnNames
from torch.utils.data import Dataset,DataLoader




def collate_fn(batch):
    # "Puts each data field into a tensor with outer dimension batch size"
    return {key: [d[key] for d in batch] for key in batch[0]}


# Generate the ground truth WHERE clause which contains all the strings used 
# corresponding to each condition

def generate_gt_where_seq( question_toks ,conds ):

    cur_values = []
    all_toks = ['<BEG>'] + question_toks + ['<END>']
    if len(conds) ==0:
        return cur_values

    for item in conds:
        
        split_tokens = unicodeToAscii(str(item[2])).split(' ')
        this_str = ['<BEG>'] + split_tokens + ['<END>']

        cur_seq = [ 
                    all_toks.index(s) if s in all_toks else 0
                    for s in this_str   
                  ]
        cur_values.append(cur_seq )

    return cur_values




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
        agg       =  sql_item_sql['agg']
        sel       =  sql_item_sql['sel']
        cond_num  =  len( sql_item_sql['conds'] ) 
        gt_cond   =  sql_item_sql['conds']
        gt_where  = generate_gt_where_seq( question_tokens, sql_item_sql['conds']  )
        
        where_col = [ x[0] for x in gt_cond ] 
        where_op  = [ x[1] for x in gt_cond ]        

        return {

            'table_id': table_id,
            'question_tokens':  question_tokens,
            'column_headers' :  column_headers,
            'column_num'     :  column_num,
            'agg'            :  agg,
            'sel'            :  sel,
            'cond_num'       :  cond_num,
            'gt_where'       :  gt_where,
            'gt_cond'        :  gt_cond,
            'where_col'      :  where_col,
            'where_op'       :  where_op,
        }

    





def train_model( model, n_epochs , optimizer,train_dataloader ,valid_dataloader,train_entry):

    model.train()

    #best_val = 2000
    
    best_agg_val = 200
    best_sel_val = 200
    best_cond_val = 200

    pred_agg , pred_sel , pred_cond = train_entry

    for e in range(n_epochs):

        epoch_loss = 0
        
        agg_val_loss = 0 
        sel_val_loss = 0
        cond_val_loss = 0



        model.train()

        for data in train_dataloader:

            model.zero_grad()
            optimizer.zero_grad()

            scores = model(data['question_tokens'] , data['column_headers'] ,train_entry , data['where_col'] ,
                        data['gt_where']
                    )

            loss = model.loss(scores,
                    
                        ( data['agg'], None , data['cond_num'] , data['where_col'] ),
                         train_entry
                     )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5 )
            optimizer.step()

        
            epoch_loss += loss.item()
        
        

        model.eval()

        for data in valid_dataloader:

           scores = model(data['question_tokens'] , data['column_headers'] ,train_entry,data['where_col'],data['gt_where'] ) 
           loss = model.validation_loss( scores,
                   (data['agg'], None,data['cond_num'] ,data['where_col'] )
                   
                   , train_entry)
        
           agg_loss , sel_loss , cond_loss = loss


           if pred_agg:
               agg_val_loss += agg_loss.item()
            
           if pred_sel:
               sel_val_loss += sel_loss.item()

           if pred_cond:
               cond_val_loss += cond_loss.item()

            
    
            #val_loss += loss.item()
        
        print('------------------------------  Epoch {} ---------------------------------\n'.format(e+1))
        print('Training loss ---------->  {}\n'.format( epoch_loss/len(train_dataloader)  )    )
        print('--------------------------------------------------------------------------\n')


        if pred_agg:

            
            epoch_agg_valid_loss = agg_val_loss/len(valid_dataloader) 
            print('\n Aggregation Model Validation Loss----------------> {}'.format(epoch_agg_valid_loss))
            if epoch_agg_valid_loss < best_agg_val:
                print('\nValidation loss decreased from {:0.6f} ------->  {:.6f}'.format(best_agg_val,epoch_agg_valid_loss) )
                print('\t Saving Model\n')
                torch.save(model.agg_predictor.state_dict() , 'saved_models/agg_predictor.pth')
                best_agg_val = epoch_agg_valid_loss

                

        if pred_sel:
            epoch_sel_valid_loss = sel_val_loss/len(valid_dataloader)
        
        
        if pred_cond:

            epoch_cond_valid_loss = cond_val_loss / len(valid_dataloader)



        print('\n------------------------------------------------------------------------\n')

        #print('  Epoch {} ----- Train Loss= {} , Valid loss= {}'.format(e+1,  epoch_loss / len(train_dataloader) , val_loss/len(valid_dataloader) ))
        


        '''
        if epoch_valid_loss < best_val:
            print('Validation Loss Decreased from {:.6f} -------->  {:.6f} '.format( best_val , epoch_valid_loss ))
            print('Saving Model ')
            torch.save(model.state_dict(), checkpoint_name)

            best_val = epoch_valid_loss
        '''


    #torch.save(model.state_dict(), 'saved_models/agg_model.pth')
    #print(model.state_dict())

        


def test_model(model,test_loader , test_entry):

    test_agg , test_sel  , test_cond = test_entry
    
    if test_agg:
        model.agg_predictor.load_state_dict( torch.load('saved_models/agg_predictor.pth')  )

    # LOADING OF STATE DICTS GOES DOWN HERE



    model.eval()
    
    agg_correct = 0
    sel_correct = 0 
    cond_correct = 0



    for data in test_loader:
        
        scores  = model(data['question_tokens'] , data['column_headers'], (True,None,None))
        
        if test_agg:

            truth = torch.from_numpy(np.asarray(data['agg']))
            out = torch.argmax( torch.exp(scores[0]),dim=1)

            res = torch.eq(truth,out)

            for i in range( len(res)):
                if res[i]:
                    agg_correct+=1

    print('\nAggregation Operator Test Accuracy =====> {}\n'.format( (agg_correct/len(test_loader.dataset))*100 ))


