# NLTODB

Final Year Btech Project on Natural Language Interface to Databases. A Natural Language interface is provided by mapping the given question analysing its meaning using Neural Networks into SQL.
Source code for the interface to this model is avaiable on https://github.com/P6A9/UI.

Try demo [here](http://34.71.161.161/)

## Team Members

* [Gayathri Krishna](https://github.com/G3Krishna)
* [Jacob James K](https://github.com/JACOBIN-SCTCS)
* [Praveen G Anand](https://github.com/P6A9)

## Implementation Details

Our work is based on the implementations derived from the following papers

* [Seq2SQL](https://arxiv.org/abs/1709.00103)
* [SQLNet](https://arxiv.org/abs/1711.04436)
* [SQLova](https://arxiv.org/pdf/1902.01069.pdf)

## Approach used

We are using [WikiSQL](https://github.com/salesforce/WikiSQL) dataset for our model. WikiSQL provides a collection of Natural Language question along with their corresponding SQL Queries tagged along with the table under consideration.The dataset is available in the data folder. Another much better but complex dataset is [Spider](https://yale-lily.github.io/spider).

We are following a sketch filling approach in which we try to fill the slots corresponding to the skeleton of a base SQL query and use that for execution in the database. The task for filling these slots are done by a combination of neural networks.

![nlttosql](https://github.com/JACOBIN-SCTCS/NLTODB/blob/master/images/nltosql.png)

The task of synthesizing the SQL query can be divided into three subtasks which are
* Predicting Aggregation Operator
* Predicting the Column to be included in the SELECT Clause
* Predicting the WHERE Clause conditions

![F](https://github.com/JACOBIN-SCTCS/NLTODB/blob/master/images/F.png)

The implementations of the first two tasks are taken from Seq2SQL while the implementation of the WHERE Clause module is taken from SQLNet.

## Steps

Pretrained models are available in saved_models folder.
Install the dependencies via
```bash
pip install -r requirements.txt
```
We  are using Pytorch for our neural network. If your device has a GPU follow instructions from pytorch.org for installing pytorch.

### Prior steps before any action
The model was tested out for working in a local system with minium level of settings so before using the model on a virtual machine please follow the steps below

* Comment out lines 31 and 32 from the file extract_vocab.py

![load_word_embed](https://github.com/JACOBIN-SCTCS/NLTODB/blob/master/images/load_word_emb.png)

* Alter the parameter use_small and set it to true to use the entire vocabulary in the glove file preprocessing.py
![use_small](https://github.com/JACOBIN-SCTCS/NLTODB/blob/master/images/use_small.png)

* Download Glove Word Vector file for use as word embeddings.
```bash
sh download_glove.sh
```
The above step will download glove.42B.300d.txt having 42B words having an embedding dimension of 300

### Training the model

**Note**: Pretrained models will be overwritten during the training process

First set the required hyperparameters of the model in train.py from lines 18-25

* **N_WORD** = It gives the dimension of the embeddings modify according to the glove file 

* **batch_size** = Set the batch size according to your need. Training was done using a batch size of 64

* **hidden_dim** = The hidden dimensions of the outputs from LSTM '

* **n_epochs**   = Number of epochs

* **train_entry** = A tuple; Set True for the modules you want to train at the same time; order is (Aggregation,Selection,Where Clause)


Training can be done by
```bash
python train.py
```

Initially the program has to load vectors into memory after which training takes place. To fasten inference we also created an API using Flask which is available in API-Word-Embeddings which can be hosted on another server and update api.py to include the address of your API server.

**Note:** Training was done without using any GPU on a VM with 4vCPU's and 15GB Ram (Loading Word Vectors need more memory). Therefore code for moving tensors into GPU and vice versa is non existent in this implementation. Tensors can be moved into GPU by checking for GPU via torch.cuda.is_available() whenever a conversion is needed and convert tensor into CUDA tensors via tensorname.cuda() and bringing it back to cpu via tensorname.cpu()


### Testing the model
Set the following hyperparameters according to your need in test.py lines 18-22
* **train_entry** = A tuple; Set True for the modules you want to test at the same time; order is (Aggregation,Selection,Where Clause)

* **N_WORD** = It gives the dimension of the embeddings modify according to the glove file used

* **batch_size** = Set the batch size according to your need. We had set the batch size as 64

* **hidden_dim** = The hidden dimensions of the outputs from LSTM 

Testing is performed via logical form accuracy in which score is only given for those outputs having a word by word match with the ground truth SQL Query.The accuracies are calculated seperately for each module.

Testing the trained model can be done by
```bash
python test.py
```
Statistics of accuracy
| Module        | Logical Form Accuracy         |
| ------------- |:-------------:|
| Aggregation Neural Network (SEQ2SQL)     | 89.5 |
| Selection Neural Network (SEQ2SQL)      | 85.4      |  
| Where Clause Neural Network (SQLNET) | 47.5      |
| Where Clause Neural Network(SQLOVA) | 49.8 |

### Inteface to the Model
Code for command line interface is available in interface.py
Modify hyperparameters available at the top of the file
You also need to modify the columns variable line 43 inorder to accomodate the info of columns in your sample table and also modifying the table name in line 22

![interface](https://github.com/JACOBIN-SCTCS/NLTODB/blob/master/images/interface.png)

The format of the columns should be given as 

```python
 [  [ 
      [col_1_token_1 , col_1_token_2 ... ,col_1_token_n ],
      [col_2_token_1 , col_2_token_2 ... ,col_2_token_n ],
      
      [col_m_token_1 , col_m_token_2 ... ,col_m_token_n ]
     ]
 ]
 
```

![col_name](https://github.com/JACOBIN-SCTCS/NLTODB/blob/master/images/col_name.png)

where m is the number of columns and each column is a list containing the words present in that column name
Example : 
```python
[  [  ['id'], ['employee','name'],['salary'] ] ]
```
After this you can interface with our model as
```bash
 python interface.py "YOUR QUESTION"
````
Example : 
```bash
 python interface.py "What is the salary of jacob"
```
The output of the program would be the corresponding SQL query to which the question is mapped. The source code for a full fledged interface using django is available in https://github.com/P6A9/UI.


## References

* https://github.com/naver/sqlova/
* https://github.com/shaabhishek/Seq2SQL
* https://github.com/xiaojunxu/SQLNet/
* https://github.com/sarim-zafar/NL2SQL-Keras








 








