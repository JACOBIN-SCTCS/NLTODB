import requests
import json
import time
import sys





def get_word_embed(word):
    
    #starttime=time.time()
    #arg=sys.argv[1]

    url="http://127.0.0.1:5000/" + word
    response = requests.get(url)

    embedding_string = response.text.split(', ')


    for x in range(1,301):
        if x == 300:
            embedding_string[300] = float(   embedding_string[x].replace("'","").replace("]","")  )
        else:
            embedding_string[x] = float( embedding_string[x].replace("'","") )


    embedding = embedding_string[1:]

    return embedding
        

#convert_to_list(response.text)
#print ("---------------%s-------------" %(time.time() - starttime))


def api_question_embed(question):

    question_embed = []
    for word in question:
        question_embed.append(get_word_embed(word))

    return question_embed

start_time = time.time()
for i in range(64):
    ret = api_question_embed(['love' ,'with','the','shape','of','you'])

print("--------------%s---------"%(time.time() - start_time))
