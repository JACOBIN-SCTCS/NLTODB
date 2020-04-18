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
                    
                         ( data['agg'], data['sel'] , data['cond_num'] , data['where_col'],
                            data['where_op'], data['gt_where'],
                         ),
                          
                          train_entry,
                     )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5 )
            optimizer.step()

        
            epoch_loss += loss.item()
        
        

        model.eval()

        for data in valid_dataloader:

           scores = model(data['question_tokens'] , data['column_headers'] ,train_entry,data['where_col'],data['gt_where'] ) 
           loss = model.validation_loss( scores,
                   (data['agg'], data['sel'],data['cond_num'] ,data['where_col'] ,
                    data['where_op'] , data['gt_where'],     
                   )
                    
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
                print('\nValidation loss decreased from {:.6f} ------->  {:.6f}'.format(best_agg_val,epoch_agg_valid_loss) )
                print('\t Saving Model\n')
                torch.save(model.agg_predictor.state_dict() , 'saved_models/agg_predictor.pth')
                best_agg_val = epoch_agg_valid_loss

                print('-------------------------------------------------------------------------\n')

        if pred_sel:
            epoch_sel_valid_loss = sel_val_loss/len(valid_dataloader)
            print('\n Selection Model Validation Loss-----------------> {}'.format(epoch_sel_valid_loss))
            if epoch_sel_valid_loss < best_sel_val:
                
                print('\nValidation loss decreased from {:.6f} ------->  {:.6f}'.format(best_sel_val,epoch_sel_valid_loss) )
                print('\t Saving Model\n')
                torch.save(model.sel_predictor.state_dict() , 'saved_models/sel_predictor.pth')
                best_sel_val = epoch_sel_valid_loss

                print('-------------------------------------------------------------------------\n')

        
        if pred_cond:

            epoch_cond_valid_loss = cond_val_loss / len(valid_dataloader)
            print('\n Conditions Prediction Model  Validation loss ------------> {}'.format(epoch_cond_valid_loss))
            if epoch_cond_valid_loss < best_cond_val:
                print('Validation loss decreased from {:.6f} -------> {:.6f}'.format(best_cond_val,epoch_cond_valid_loss))
                print('\t Saving Model\n')
                torch.save(model.cond_predictor.state_dict(),'saved_models/cond_predictor.pth')
                best_cond_val = epoch_cond_valid_loss

                print('-------------------------------------------------------------------------\n')


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



def check_accuracy(pred_cond, gt_cond):

    correct = 0

    num_err = 0 
    col_err = 0
    op_err  = 0
    str_err = 0
    
    
    
    for b in range(len(pred_cond)):
        
        flag = True

        if len(pred_cond[b]) != len(gt_cond[b]):
            flag = False
            num_err += 1 

        if flag and set(x[0] for x in pred_cond[b]) != set(y[0] for y in gt_cond[b]):

            flag = False
            col_err += 1 

        for idx in range( len(pred_cond[b]) ):

            if not flag:
                break

            gt_idx = tuple(x[0] for x in gt_cond[b]).index(pred_cond[b][idx][0])

            if flag and gt_cond[b][gt_idx][1] != pred_cond[b][idx][1]:
                flag = False
                op_err += 1

        for idx in range(len(pred_cond[b])):
            if not flag:
                break

            gt_idx = tuple(x[0] for x in gt_cond[b]).index(pred_cond[b][idx][0])

            if flag and str(gt_cond[b][gt_idx][2]).lower() != str(pred_cond[b][gt_idx][2]).lower():
                flag = False
                str_err += 1


        if flag==True:
            correct+=1
        
    return (num_err,col_err,op_err,str_err,correct)



# Exact code from Xiaojunxu SQLnet repo for ensuring additional safety
# when predicting the strings in the WHERE Clause

def merge_tokens(tok_list , raw_tok_str):

    tok_str  = raw_tok_str.lower()
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
    special  = { 
                    '-LRB-':'(',
                    '-RRB-':')',
                    '-LSB-':'[',
                    '-RSB-':']',
                    '``':'"',
                    '\'\'':'"',
                    '--':u'\u2013'
                }
    ret = ''
    double_quote_appear = 0
    
    for raw_tok in tok_list:
        
        if not raw_tok:
            continue
        
        tok = special.get(raw_tok,raw_tok)
        if tok == '"':
            double_quote_appear = 1 - double_quote_appear

        if len(ret) == 0:
            pass
        
        elif len(ret)>0  and ret +' '+ tok in tok_str:
            ret = ret+ ' '
        
        elif len(ret)>0 and ret+tok   in tok_str:
            pass
        elif tok =='"':
            if double_quote_appear :
                ret = ret + ' '
        elif tok[0] not in alphabet:
            pass
        elif ( ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']  ) and (ret[-1] !='"' or not double_quote_appear):

            ret = ret + ' '

        ret = ret+tok

    return ret.strip()



def gen_query_acc( cond_scores, questions ):

    cond_num_score, cond_col_score, cond_op_score, cond_str_score = [
                x.data.cpu().numpy()  for x in cond_scores
            ]

    pred_cond = [] 

    for b in range(len(cond_num_score)):

        b_cond = []
        cond_num = np.argmax(cond_num_score[b])

        all_toks = ['<BEG>'] + questions[b] + ['<END>']
        max_idxes = np.argsort(-cond_col_score[b])[:cond_num]

        for i in range(cond_num):
            cur_cond = [] 
            cur_cond.append(max_idxes[i])
            cur_cond.append(np.argmax(cond_op_score[b][i]))
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







def test_model(model,test_loader , test_entry):

    test_agg , test_sel  , test_cond = test_entry
    
    if test_agg:
        model.agg_predictor.load_state_dict( torch.load('saved_models/agg_predictor.pth')  )

    if test_sel:
        model.sel_predictor.load_state_dict( torch.load('saved_models/sel_predictor.pth')  )

    if test_cond:
        model.cond_predictor.load_state_dict(torch.load('saved_models/cond_predictor.pth'))



    # LOADING OF STATE DICTS GOES DOWN HERE



    model.eval()
    
    agg_correct = 0
    sel_correct = 0 


    cond_correct = 0
    cond_num_err = 0
    cond_col_err = 0
    cond_op_err  = 0 
    cond_str_err = 0


    for data in test_loader:
        
        scores  = model(data['question_tokens'] , data['column_headers'],test_entry,
                  data['where_col'], data['gt_where']
                )
        
        if test_agg:

            truth = torch.from_numpy(np.asarray(data['agg']))
            out = torch.argmax( torch.exp(scores[0]),dim=1)

            res = torch.eq(truth,out)

            for i in range( len(res)):
                if res[i]:
                    agg_correct+=1

        if test_sel:
            
            truth_sel = torch.from_numpy(np.asarray(data['sel']))
            out_sel = torch.argmax( torch.exp(scores[1]),dim=1)
            
            res_sel = torch.eq(truth_sel,out_sel)

            for i in range(len(res_sel)):
                if res_sel[i]:
                    sel_correct+=1

        if test_cond:
            
            pred_cond = gen_query_acc( scores[2], data['question_tokens']  )
            
            a,b,c,d,e = check_accuracy(pred_cond , data['gt_cond'])

            cond_num_err += a
            cond_col_err += b
            cond_op_err  += c
            cond_str_err += d
            cond_correct += e

    if test_agg:


        print('\nAggregation Operator Test Accuracy =====> {}\n'.format( (agg_correct/len(test_loader.dataset))*100 ))
    
    if test_sel:
        
        print('\n Selection Operation Test Accuracy =====> {}\n'.format((sel_correct/len(test_loader.dataset))*100  ))

    if test_cond:

        #length = len(test_loader.dataset)
        
        print('\n Condition Predictor Test Accuracy======>{}\n'.format( ( cond_correct /len(test_loader.dataset))*100  ))
       
        #print( 'Condtion Number accuracy' +  str ( (length-cond_num_err)/length *100)   ) 
        #print( 'Condtion Column accuracy' +  str ( (length-cond_col_err)/length *100)   )
        #print( 'Condtion Number accuracy' +  str ( (length-cond_op_err)/length *100)   )
        #print( 'Condtion string accuracy' +  str ( (length-cond_str_err)/length *100)   )



def gen_sql_query(agg_idx,sel_idx,conds,cols,table_name,symbol='_'):

    aggs = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=','<','>']

    def merge_column_name(column_name,symbol):
        
        return symbol.join(column_name)

    def generate_each_cond(cols,condition,symbol):
        
        s = ''
        s = s+ merge_column_name( cols[condition[0]],symbol ) + ' ' + cond_ops[condition[1]] + ' \"' + condition[2] + '\"'

        return s



    query = ''
    if agg_idx >0:
        query = query+ '\nSELECT '+aggs[agg_idx]+'( '+ merge_column_name(cols[sel_idx],symbol) + ' )\n'
    else:
        query = query + '\nSELECT '+ merge_column_name(cols[sel_idx],symbol) +'\n'

    if len(conds)==0:

        query = query + 'FROM ' + table_name + ';\n'
        return query

    else:

        query = query + 'FROM ' + table_name + '\nWHERE '

        conditions_list = []
        for cond in conds:
            conditions_list.append(generate_each_cond(cols,cond,symbol))

        cond_str = ' AND '.join(conditions_list)

        query = query + cond_str +' ;\n'
        return query
