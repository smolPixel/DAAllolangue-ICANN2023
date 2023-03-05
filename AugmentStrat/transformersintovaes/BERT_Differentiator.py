from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from process_data import *
from sklearn.metrics import accuracy_score
import os, io
import numpy as np
import json
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from nltk.tokenize import TweetTokenizer, sent_tokenize
from torchtext.vocab import build_vocab_from_iterator
import math
import ast
import pickle
import torch
import copy
import random




def initialize_dataset_from_dataframe(train, dev):
    # train=dataframe
    allsentences = list(train['sentence'])

    tokenizer = TweetTokenizer()
    allsentences = [tokenizer.tokenize(sentence) for sentence in allsentences if sentence == sentence]
    # print(allsentences)
    vocab = build_vocab_from_iterator(allsentences, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    train = ds_DAControlled(train, tokenizer, vocab)
    dev = ds_DAControlled(dev, tokenizer, vocab, dev=True)
    return train, dev


class ds_DAControlled(Dataset):


    def __init__(self, data, tokenizer=None, vocab=None, argdict=None, dev=False, dataset_parent=None, from_dict=False):
        super().__init__()
        """data: tsv of the data
           tokenizer: tokenizer trained
           vocabInput+Output: vocab trained on train"""
        self.argdict=None
        if from_dict:
            self.data=data
            self.max_len=dataset_parent.max_len
            self.vocab=dataset_parent.vocab
            self.tokenizer=dataset_parent.tokenizer
        else:
            self.data = {}
            self.max_len = 0
            if self.max_len==0:
                find_max_len=True
            else:
                find_max_len=False
            self.vocab = vocab
            self.tokenizer=tokenizer
            # self.pad_idx = self.vocab['<pad>']
            self.max_len_label=0
            self.max_len_words=0
            self.num_sentences=0
            self.len_sentence=0
            index=0
            for i, row in data.iterrows():
                #For sentence VAE
                # sentences_sep=row['sentences'].strip().split('.')
                # print(sentences_sep)
                # sentences_sep=[sent.replace('.',' . ') for sent in sent_tokenize(row['sentence'].strip().replace(' . ', '.'))]
                #TODO CHECK IF THIS HERE DOES SOMETHING IMPORTANT
                sentences_sep=[vocab(self.tokenizer.tokenize("<bos> "+sent+" <eos>")) for sent in row['sentence']]
                if len(sentences_sep)>self.num_sentences:
                    self.num_sentences=len(sentences_sep)
                # if row['sentence'] in ['.', '']:
                #     continue
                if self.len_sentence<max([len(sent) for sent in sentences_sep]):
                    self.len_sentence=max([len(sent) for sent in sentences_sep])
                tokenized_text = self.tokenizer.tokenize("<bos> " + row['sentence'] + " <eos>")
                if find_max_len and self.max_len<len(tokenized_text):
                    self.max_len=len(tokenized_text)
                input = np.array(vocab(tokenized_text))
                # tokenized_text=self.tokenizer.tokenize("<bos> "+row['sentence']+" <eos>")
                # sentence_max_len=" ".join(row['sentences'].split(' ')[:self.max_len])
                # output=np.array(vocab(tokenized_labels))
                # if len(output)>self.max_len_label:
                #     self.max_len_label=len(output)
                self.data[index] = {'input': input, 'label':row['label'], 'sentence':row['sentence'], 'augmented':False}
                index+=1

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eos_idx(self):
        return self.vocab['<eos>']

    @property
    def pad_idx(self):
        return self.vocab['<pad>']

    @property
    def bos_idx(self):
        return self.vocab['<bos>']

    @property
    def unk_idx(self):
        return self.vocab['<unk>']

    def get_i2w(self):
        return self.vocab.get_itos()

    def __setitem__(self, key, value):
        self.data[key]=value

    def reset_index(self):
        new_dat={}
        for i, (j, dat) in enumerate(self.data.items()):
            new_dat[i] = dat
        self.data=new_dat

    def get_random_example_from_class(self, classe):
        lab=None
        while lab!=classe:
            random_ex=random.randint(0, len(self.data)-1)
            dat=self.data[random_ex]
            lab=dat['label']
        return dat


    def tokenize(self, sentence):
        "Tokenize a sentence"
        return self.vocab(self.tokenizer.tokenize("<bos> "+sentence+" <eos>"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input = self.data[item]['input'][:self.max_len]
        label = self.data[item]['label']
        input=np.append(input, [self.pad_idx] * (self.max_len - len(input))).astype(int)
        return {
            'index': item,
            'input': input,
            'label': label,
            'sentence':self.data[item]['sentence'],
            'augmented':self.data[item]['augmented'],
    }



class Bert_Differentiator():

	def __init__(self):
		max_seq_len=64
		pass
		# print(self.model)


	def run_epoch(self, train, model, tokenizer, optimizer):

	    data_loader = DataLoader(
	        dataset=train,
	        batch_size=32,
	        shuffle=True,
	        # num_workers=cpu_count(),
	        pin_memory=torch.cuda.is_available()
	    )

	    for i, batch in enumerate(data_loader):
	        optimizer.zero_grad()
	        # print(batch)
	        # fds
	        text_batch=batch['sentence']
	        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=64)
	        input_ids = encoding['input_ids'].cuda()
	        attention_mask = encoding['attention_mask'].cuda()
	        # print(encoding)
	        labels=batch['label'].cuda()
	        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	        # print(outputs)
	        # print(loss, outputs)
	        # print(outputs)
	        loss = outputs.loss
	        loss.backward()
	        optimizer.step()

	    #Test
	    return 0

	def calculateAccuracyDev(self, dev, model, tokenizer):
	    data_loader = DataLoader(
	        dataset=dev,
	        batch_size=64,
	        shuffle=False,
	        # num_workers=cpu_count(),
	        pin_memory=torch.cuda.is_available()
	    )
	    pred = torch.zeros(len(dev))
	    Y = torch.zeros(len(dev))
	    start = 0
	    for i, batch in enumerate(data_loader):
	        with torch.no_grad():
	            text_batch = batch['sentence']
	            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=64)
	            input_ids = encoding['input_ids'].cuda()
	            attention_mask = encoding['attention_mask'].cuda()
	            # print(encoding)
	            labels = batch['label'].cuda()
	            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	            results = torch.argmax(torch.log_softmax(outputs.logits, dim=1), dim=1)
	            pred[start:start + 64] = results
	            Y[start:start + 64] = batch['label']
	            start = start + 64
	    return accuracy_score(Y, pred)

	def train(self):
		# self.tokenizer = BertTokenizer.from_pretrained('Models/bert_labellers_tokenizer.ptf')
		# train=pd.read_csv('train_differentiator.tsv', sep='\t', index_col=0)
		# dev=pd.read_csv('dev_differentiator.tsv', sep='\t', index_col=0)
		# train, dev=initialize_dataset_from_dataframe(train, dev)
		# self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()
		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
		# for i in range(4):
		# 	self.run_epoch(train, self.model, self.tokenizer, self.optimizer)
		# 	acc=self.calculateAccuracyDev(dev, self.model, self.tokenizer)
		# 	print(i, acc)
		# self.model.save_pretrained('Models/bert_differentiator')
		# fds
		self.model = BertForSequenceClassification.from_pretrained('AugmentStrat/transformersintovaes/Models/bert_differentiator').cuda()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# acc = self.calculateAccuracyDev(dev, self.model, self.tokenizer)
		# print("Model has already been trained")

	def label(self, sentences):
		# model = BertForSequenceClassification.from_pretrained('Models/bert_labeller').cuda()
		# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		pred = torch.zeros(len(sentences))
		probs = torch.zeros(len(sentences))
		# Y = torch.zeros(len(dev))
		start = 0
		for i in range(0, len(sentences), 64):
			with torch.no_grad():
				text_batch = sentences[start:start+64]
				encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
				input_ids = encoding['input_ids'].cuda()
				attention_mask = encoding['attention_mask'].cuda()
				# print(encoding)
				# print(encoding)
				labels = torch.zeros(len(sentences)).long()
				outputs = self.model(input_ids, attention_mask=attention_mask)
				results = torch.argmax(torch.log_softmax(outputs.logits, dim=1), dim=1)
				pred[start:start + 64] = results
				# print(torch.max(torch.softmax(outputs.logits, dim=1), dim=1).indices)
				probs[start:start+64]=torch.max(torch.softmax(outputs.logits, dim=1), dim=1).values
				# print(prob)
				start = start + 64

		# print(probs)
		return pred, probs

bb=Bert_Differentiator()
bb.train()