import time
import sys

starttime=time.time()

f= open("glove.42B.300d.txt", "r")

param=sys.argv[1]
word=" "
for lines in f:
	words=lines.split()
	for i in words:
		if i.isalpha()==True and i==param:
			word= words[0:]
			break
print (word)
print ("---------------%s-------------" %(time.time() - starttime))
