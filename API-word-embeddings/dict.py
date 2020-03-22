import numpy
import time
from json import JSONEncoder
import json


class NumpyArrayEncoder(JSONEncoder):
	def default(self,obj):
		if isinstance(obj,numpy.ndarray):
			return obj.tolist()
		return JSONEncoder.default(self,obj)

start=time.time()
ret={}
document=open("glove.6B.300d.txt", "r")
doc=open("dict.txt","w")
for line in document:
	info=line.strip().split(' ')
	if info[0].lower not in ret:
		ret[info[0].lower()]=numpy.array(tuple(round(float(x),2) for x in info[1:]))

json.dump(ret, doc, cls=NumpyArrayEncoder)
doc.close()
document.close()

print("----------------%s----------------" %(time.time()-start))
