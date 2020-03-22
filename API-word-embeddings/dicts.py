import time
import json
import sys

start=time.time()

file=open("dict.txt","r")

ret=json.load(file)

arg=sys.argv[1]
for key, value in ret.items():
	if arg==key:
		print (value)
		break

print("----------------%s----------------" %(time.time()-start))
