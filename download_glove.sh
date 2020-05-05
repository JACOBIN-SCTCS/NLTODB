#To download the Glove word embedding tokens. 
#Script written for testing on local system.

# Reference https://github.com/xiaojunxu/SQLNet/blob/master/download_glove.sh


if [[ ! -d glove ]]; then
	mkdir glove
fi

cd glove
wget http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip
