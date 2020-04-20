import requests
import json
import time
import sys
import numpy as np



# Contributed by https://github.com/P6A9

def get_word_embed(word):
    
    #starttime=time.time()
    #arg=sys.argv[1]
    if word=="#":
        url="http://35.184.36.137/embed?id=%23"
    else:
        url="http://35.184.36.137/embed?id=" + word
    response = requests.get(url)

    if (response.text==word):

        return np.zeros(300,dtype=np.float32 )

    embedding_string = response.text.split(' ')
    
    embedding = [float(0) for _ in range(300)] 

    for x in range(1,len(embedding_string)):
        if '\n' in embedding_string[x]:
            embedding[(x-1)] = float(   embedding_string[x].replace('\n',''))
          
        else:
            embedding[(x-1)] = float( embedding_string[x] )
        
        
        
        
        #if x == 300:
        #    embedding_string[300] = float(   embedding_string[x].replace('\n',''))
        #else:
        #    embedding_string[x] = float( embedding_string[x] )


    #embedding = embedding_string[1:]
    return embedding


#convert_to_list(response.text)
#print ("---------------%s-------------" %(time.time() - starttime))


def api_question_embed(question):

    question_embed = []
    for word in question:
        if word.isdigit():
            total_sum = np.zeros(300,dtype=np.float32 )
            for digit in word:
                total_sum += get_word_embed(str(digit))
            
            question_embed.append(total_sum/len(word))


        else:

            question_embed.append(get_word_embed(word))

    return question_embed



