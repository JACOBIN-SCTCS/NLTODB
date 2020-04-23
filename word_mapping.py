
import string

word_mappings = {


                    "doesn't" : [ "doesn't" ], 
                    "can't"   : [ "can't"  ],
                    "won't"   : ["would","not"] ,
                    "don't"   : ["don't"    ],
                    "i've"    : ["i've" ],
                    "i'd"     : ["i'd" ],
                    "i'm"     : ["i'm" ] ,
                    "i'll"    : ["i'll"],
                    "she's"   : ["she's"],
                    "he's"    : ["he's"],
                    "it's"    : ["it's"],
                    "there's" : ["there's"],
                    "they're" : ["they" , "are" ] ,
                    "we're"   : ["we","are"],
                    "you've"  : ["you've"],
                    "you're"  : ["you're"],
                    "couldn't": ["could","not"],
                    "shouldn't": ["should","not"],
                    "wouldn't" : ["would", "not"],

                }


def process_sentence(sentence):

    sentence = sentence.lower()
    
    new_sentence = []
     
    for word in sentence.split(' '):
        
        if word == '':
            continue
        if word in string.punctuation:
            continue
        if '?' in word:
            word=word.replace('?','')

        if '\'' in word:

            if word in word_mappings.keys():
                for each in word_mappings[word]:
                    new_sentence.append(each)
            else:

                index = word.index('\'')
                word = word[:index]
                new_sentence.append(word)
        
        elif '%' in word:
            new_sentence.append( word.replace("%","") )

        else:
            new_sentence.append(word)

    
    return ' '.join(new_sentence)


