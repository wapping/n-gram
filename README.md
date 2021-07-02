# About
A simple implementation of N-gram language model.

# Requirements
- numpy

# Data preparation
## Corpus
Training data for the N-gram model, a text file like this:
```
曼联加油
懂球直播
有也免费高清的额
直播挺全的
曼联这局肯定胜利
```
Text lines will be split into tokens by a delimiter when training. By default, no delimiter given, text lines will be split into characters.

## Tokens
The dictionary for the model, a text file, each line of which is a token. Every token is unique in the file. 
```
光
衰
戒
颅
阖
```
# Training
Run the script `train_n_gram.py` to train an N-gram model. 

```    
python train_n_gram.py --corpus_path data/tieba.dialogues --token_path data/charset.txt --model_path data/2-gram.model --n 2
```

# Testing
Run the script `test_n_gram.py` to test the trained N-gram model. 

```    
python test_n_gram.py --token_path data/charset.txt --model_path data/2-gram.model --text 哈哈
```
The testing output will like:
```
INFO - Loaded model from data/2-gram.model
INFO - Model info:
	n: 2
	head2tail length: 5947
	tokens: 5952
The most probable next token of the '哈哈' is '哈'.
```
