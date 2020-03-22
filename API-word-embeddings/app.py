from flask import Flask

app=Flask(__name__)



@app.route('/<string:param>')
def ikea(param):
	return check(param)
	
def check(param):
	document=open("/home/projectf31/NLTODB/glove/glove.42B.300d.txt","r")
	word=param
	for lines in document:
		words=lines.split()
		if words[0]==param:
			word= str(words[0:])
			break
	return word
if __name__== "__main__":
	app.run(debug=True)

