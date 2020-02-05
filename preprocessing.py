import unicodedata




# The function below takes in the the type of the file (test/validation/trainset) and
# returns the contents of the file suitable for processing


def load_dataset(filetype,use_small=True):
    
    sql_file_name = 'data/'+filetype+'.jsonl'
    tables_file_name = 'data/'+filetype+'.tables.jsonl'


    # Gets the dataset contents and load it to a variable

    sql_file = open(sql_file_name,'r')
    tables_file = open(tables_file_name,'r')

    tables_list = tables_file.readlines()

    sql_list = []
    tables_dict = {} 

    for table in tables_list:
        table = eval(table)

        tables_dict[table['id']] = table
    
    if use_small:

        for dataobj in sql_file.readlines()[:1000]:
        
            # eval() function is used to convert a dictionary represented in string to python dictionary
            dataobj = eval(dataobj)
            sql_list.append(dataobj)
    else:

        for dataobj in sql_file.readlines():
            dataobj = eval(dataobj)
            sql_list.append(dataobj)
        
        
    return sql_list,tables_dict



# The function below is used to normalize a string consisting of a mixture of characters
# to ASCII using the unicodedata.normalize function
#   Taken from https://www.programcreek.com/python/example/1020/unicodedata.category
#   and https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#


def unicodeToAscii(text):

    suffix_s =  [ ('how\'s','how is') , ('what\'s','what is') ] 

    text = text.lower()
    text = text.replace('?','')
    text = text.replace('\u00a0',' ')
    text = text.replace('\"','')
    for suffix in suffix_s:
        text = text.replace(suffix[0],suffix[1])

    
    text = unicodedata.normalize("NFD",text)
    output = []
    for char in text:
        if unicodedata.category(char)!='Mn':
            output.append(char)
    return "".join(output)




# The function takes in a list of column names and produce a list of list of tokens in each column
# Example :   ['Time', 'Big Ten Team']     ------>    [ ['time'] , ['big','ten','team'], ] 
#

def splitColumnNames(column_list):
    return [ unicodeToAscii(col).split(' ') for col in column_list ]











#print( len(splitColumnNames(['Time', 'Big Ten Team'])))

    
    
    



