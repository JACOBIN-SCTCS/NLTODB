import unicodedata




# The function below takes in the the type of the file (test/validation/trainset) and
# returns the contents of the file suitable for processing


def load_dataset(filetype):
    
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

        tables_dict[table['id']] = table['header']
    
    for dataobj in sql_file.readlines()[:1000]:
        
        # eval() function is used to convert a dictionary represented in string to python dictionary
        dataobj = eval(dataobj)
        sql_list.append(dataobj)
            
        
    return sql_list,tables_dict



# The function below is used to normalize a string consisting of a mixture of characters
# to ASCII using the unicodedata.normalize function
#   Taken from https://www.programcreek.com/python/example/1020/unicodedata.category
#   and https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#


def unicodeToAscii(text):

    text = unicodedata.normalize("NFD",text)
    output = []
    for char in text:
        if unicodedata.category(char)!='Mn':
            output.append(char)
    return "".join(output)







