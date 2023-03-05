import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
#implementation of GDaugC-Influence as presented in https://arxiv.org/pdf/2004.11546.pdf
import torch.autograd as autograd
import numpy as np
import copy

def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam

import torch.autograd as autograd
from scipy import stats
#
# class MyBertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config, num_labels):
#         super(MyBertForSequenceClassification, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits

class influenceFunction_metastrat():

	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.argdict=argdict
		self.augmentator=augmentator

	def influence(self, train, dev, test, classifier, HVP):
		results_full = torch.zeros((len(dev)))
		set_seed(7)
		classifier.init_model()
		classifier.train_test(train, dev, test, None)
		data_loader = DataLoader(
			dataset=dev,
			batch_size=1,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		for j, batch in enumerate(data_loader):
			label=batch['label']
			with torch.no_grad():
				loss = classifier.get_loss(batch).item()
				pred=classifier.get_logits(batch)
				pred=torch.argmax(torch.softmax(pred, dim=-1))
				print(pred.item())
				print(label)
				fds
			results_full[j] = loss

		matrix_results = torch.zeros((len(train), len(dev)))


		#As a first step let's find the influence function of a dev ex
		#Param influence
		index=1

		classifier.train_test(train, dev, None, None)
		# ex_test=dev.data[0]
		ex_test=train.data[1]
		print('-----')
		print(ex_test)
		print('-----')
		data_loader = DataLoader(
			dataset=dev,
			batch_size=1,
			shuffle=False,
			drop_last=True,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		for index_dev, batch_dev in enumerate(data_loader):
			set_seed(7)
			classifier.init_model()
		#
		# model = classifier.algo.model
		# tokenizer= classifier.algo.tokenizer
		#
		# #test decision
		# tokenized_test=tokenizer(ex_test['sentence'], return_tensors='pt')
		# output_test=model(tokenized_test['input_ids'].cuda(), attention_mask=tokenized_test['attention_mask'].cuda())['logits']
		# data_loader = DataLoader(
		# 	dataset=train,
		# 	batch_size=1,
		# 	shuffle=False,
		# 	drop_last=True,
		# 	# num_workers=cpu_count(),
		# 	pin_memory=torch.cuda.is_available()
		# )
		# for bs in data_loader:
		# 	output_test=classifier.get_logits(bs)
		# 	break
		#
		# print(output_test)
		# decision=torch.argmax(torch.softmax(output_test, dim=-1))
		# if decision.item()==ex_test['label']:
		# 	print("correct")
		# else:
		# 	print(decision, ex_test['label'])
		# tokenizer = BertTokenizer.from_pretrained('', do_lower_case=True)

		# Prepare model
		# model = MyBertForSequenceClassification.from_pretrained('xhan/influence-function-analysis/SA_tagger_output_bert_e3', num_labels=2).to('cuda')

			frozen = ['bert.embeddings.',
					  'bert.encoder.layer.0.',
					  'bert.encoder.layer.1.',
					  'bert.encoder.layer.2.',
					  'bert.encoder.layer.3.',
					  'bert.encoder.layer.4.',
					  'bert.encoder.layer.5.',
					  'bert.encoder.layer.6.',
					  'bert.encoder.layer.7.',
					  'embedding.weight'
					  ]
			param_influence=[]
			for n, p in list(classifier.algo.named_parameters()):
				if (not any(fr in n for fr in frozen)):
					param_influence.append(p)
				elif 'bert.embeddings.word_embeddings.' in n:
					pass
				else:
					p.requires_grad=False
			# sent=ex_test['sentence']
			# label=torch.tensor(dev.data[0]['label']).cuda()
			# encoding=tokenizer(sent, return_tensors='pt')
			# # print(encoding)
			# input_ids = encoding['input_ids'].cuda()
			# attention_mask = encoding['attention_mask'].cuda()
			# tokenized=tokenizer(sent)
			# outputs = model(input_ids, attention_mask=attention_mask)
			# # print(outputs)
			# # logits=outputs.logits
			# logits=outputs

			classifier.algo.zero_grad()
			test_loss=classifier.get_loss(batch_dev)
			# test_loss=model(input_ids, attention_mask=attention_mask, labels=label).loss
			# test_loss=model(input_ids, attention_mask=attention_mask, labels=label).loss
			test_grad=autograd.grad(test_loss, param_influence)

			#Compute HVP-1
			data_loader = DataLoader(
				dataset=train,
				batch_size=8,
				shuffle=False,
				drop_last=True,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			classifier.algo.train()
			inverse_hvp=self.get_inverse_hvp_lissa(test_grad, classifier, 'cuda', param_influence, data_loader, damping=0, num_samples=1, recursion_depth=int(len(train)))
			influences = np.zeros(len(train))
			train_tok_sal_lists = []

			data_loader = DataLoader(
				dataset=train,
				batch_size=1,
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)

			for i, batch in enumerate(tqdm(data_loader, desc="Train set index")):
				classifier.algo.train()
				# tokenized = tokenizer(batch['sentence'], return_tensors='pt', padding=True)
				# input_ids = tokenized['input_ids'].cuda()
				# attention_mask = tokenized['attention_mask'].cuda()
				# label_ids = torch.tensor(batch['label']).cuda()


				######## L_TRAIN GRADIENT ########
				classifier.algo.zero_grad()
				train_loss=classifier.get_loss(batch)
				# train_loss = model(input_ids, attention_mask=attention_mask, labels=label_ids).loss
				train_grads = autograd.grad(train_loss, param_influence)
				influences[i] = torch.dot(inverse_hvp, self.gather_flat_grad(train_grads)).item()
				# if i==index:
				# 	print("------")
				# 	print(batch)
				# 	print("------")

			order_of_interest = "max"
			label_of_interest = "both"
			num_of_interest = 50

			# print("----")
			# print(influences.shape)
			# print(influences)
			results_full[index_dev].item()
			influences=influences-results_full[index_dev].item()
			# print(results_full[index_dev])
			# print(influences)
			# print(matrix_results.shape)
			# print(results_full.shape)
			# fds
			matrix_results[:, index_dev]=torch.Tensor(influences)
			# print(matrix_results)
			# fds

		torch.save(matrix_results, "Influence.pt")
			# fds
			# print("----")
			#
			# train_idx_sorted = list(np.argsort(influences))
			# if order_of_interest == "max":
			# 	train_idx_sorted.reverse()
			# if label_of_interest == "both":
			# 	valid_labels = ["0", "1"]
			# else:
			# 	valid_labels = [label_of_interest]
			#
			# cnt = 0
			# for idx in train_idx_sorted:
			# 	te = train.data[idx]
			# 	print(te['sentence'], te['label'])
			# 	print(influences[idx])
			# 	cnt += 1
			# 	if cnt >= num_of_interest:
			# 		break
			# fds
			#
			# grads=self.get_val_grad(dev, model)
			# HPV=get_HPV(train, model, grads)
			#
			# fds

	def gather_flat_grad(self, grads):
		views = []
		for p in grads:
			if p.data.is_sparse:
				view = p.data.to_dense().view(-1)
			else:
				view = p.data.view(-1)
			views.append(view)
		return torch.cat(views, 0)

	def hv(self, loss, model_params, v):  # according to pytorch issue #24004
		#CHECK OF HV FCUNTION: OK!
		#     s = time.time()
		# for obj in model_params:
		# 	print(obj.requires_grad)
		# print(loss.loss)
		# print(autograd.grad(loss, model_params[0]))
		grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
		# print(grad)
		# fds
		#     e1 = time.time()
		Hv = autograd.grad(grad, model_params, grad_outputs=v)
		#     e2 = time.time()
		#     print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
		return Hv

	def get_inverse_hvp_lissa(self, v, classifier, device, param_influence, train_loader, damping, num_samples, recursion_depth,
							  scale=1e4):
		damping=0.003
		recursion_depth=1730
		ihvp = None
		for i in range(num_samples):
			cur_estimate = v
			lissa_data_iterator = iter(train_loader)
			for j in range(recursion_depth):
				try:
					batch = next(lissa_data_iterator)
					# tokenized = tokenizer(batch['sentence'], return_tensors='pt', padding=True)
					# input_ids = tokenized['input_ids']
					# attention_mask = tokenized['attention_mask']
					# label_ids = torch.tensor(batch['label']).cuda()
				except StopIteration:
					lissa_data_iterator = iter(train_loader)
					batch = next(lissa_data_iterator)
					# tokenized = tokenizer(batch['sentence'], return_tensors='pt', padding=True)
					# input_ids = tokenized['input_ids']
					# attention_mask = tokenized['attention_mask']
					# label_ids = torch.tensor(batch['label'])
				# input_ids = input_ids.to(device)
				# input_mask = attention_mask.to(device)
				# label_ids = label_ids.to(device)
				classifier.algo.zero_grad()
				# train_loss = model(input_ids, attention_mask=input_mask, labels=label_ids).loss
				train_loss=classifier.get_loss(batch)
				# train_loss = model(input_ids, attention_mask=input_mask, labels=label_ids).loss
				hvp = self.hv(train_loss, param_influence, cur_estimate)
				# print(hvp)
				# fd
				cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
				if (j % 200 == 0) or (j == recursion_depth - 1):
					print("Recursion at depth %s: norm is %f" % (
					j, np.linalg.norm(self.gather_flat_grad(cur_estimate).cpu().numpy())))
			if ihvp == None:
				ihvp = [_a / scale for _a in cur_estimate]
			else:
				ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
		return_ihvp = self.gather_flat_grad(ihvp)
		return_ihvp /= num_samples
		return return_ihvp

	def get_HPV(self, train, classifier, v):
		model=classifier.algo.model
		tokenizer=classifier.algo.tokenizer
		for r in range(5):
			res=[w.clone().cuda() for w in v]
			model.zero_grad()
			data_loader = DataLoader(
				dataset=train,
				batch_size=32,
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)

			model = classifier.algo.model
			tokenizer = classifier.algo.tokenizer
			# loss_function = torch.nn.functional.cross_entropy
			model.zero_grad()
			# Accumulate gradient over all exos
			for batch in tqdm(data_loader, desc="Calculating HVP"):
				model.eval()
				text_batch = batch['sentence']
				encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
				input_ids = encoding['input_ids'].cuda()
				attention_mask = encoding['attention_mask'].cuda()
				# print(encoding)
				labels = batch['label'].cuda()
				outputs = model(input_ids, attention_mask=attention_mask)
				logits = outputs.logits
				# We are interested in cross entropy loss
				loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
				grad_list = torch.autograd.grad(loss,
												model.parameters(),
												create_graph=True)
				grad=[]
				H=0
				for i, (g, g_v) in enumerate(zip(grad_list, res)):
					H+=(g*g_v).sum()
				#H=grad@v
				H.backward()
				# print(res[20])
				gfsd

		pass

	def get_val_grad(self, dev, classifier):
		data_loader = DataLoader(
			dataset=dev,
			batch_size=32,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		model=classifier.algo.model
		tokenizer=classifier.algo.tokenizer
		# loss_function = torch.nn.functional.cross_entropy
		model.zero_grad()
		#Accumulate gradient over all exos
		for batch in tqdm(data_loader, desc="Calculating validation grad"):
			model.eval()
			text_batch=batch['sentence']
			encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
			input_ids = encoding['input_ids'].cuda()
			attention_mask = encoding['attention_mask'].cuda()
			# print(encoding)
			labels = batch['label'].cuda()
			outputs = model(input_ids, attention_mask=attention_mask)
			logits=outputs.logits
			#We are interested in cross entropy loss
			loss=torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
			loss.backward()
			# loss=loss_function(logits, labels)

		grad=[]
		for p in model.parameters():
			if p.grad is None:
				print("wrong")
				fds
			grad.append((p.grad.data/len(dev)).cpu())

		return grad


	def loo(self, train, dev, classifier):
		#Leave one out retraining
		print(len(train))
		print(len(dev))
		#Get OG results
		results_full=torch.zeros((len(dev)))
		set_seed(7)
		classifier.init_model()
		classifier.train_test(train, dev)
		data_loader = DataLoader(
			dataset=dev,
			batch_size=1,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		for j, batch in enumerate(data_loader):
			with torch.no_grad():
				loss = classifier.get_loss(batch).item()
			results_full[j] = loss

		matrix_results=torch.zeros((len(train), len(dev)))
		for i in range(len(train)):
			training_new=copy.deepcopy(train)
			training_new.data.pop(i)
			training_new.reset_index()
			print(len(training_new))
			set_seed(7)
			classifier.init_model()
			classifier.train_test(training_new, dev)
			data_loader = DataLoader(
				dataset=dev,
				batch_size=1,
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			for j, batch in enumerate(data_loader):
				with torch.no_grad():
					loss=classifier.get_loss(batch).item()
				#Note the impact on the loss of removing this training data for each data point
				matrix_results[i, j]=loss-results_full[j]
		print(matrix_results)
		torch.save(matrix_results, "LeaveOneOutLinear.pt")

		fds

	def augment(self, train, dev, test):
		from Classifier.classifier import classifier
		classifier=classifier(argdict=self.argdict, train=train)
		# v_init=self.get_val_grad(dev, classifier)
		# H=self.get_HPV(train, classifier, v_init)
		# fds

		# self.loo(train, dev, classifier)
		# fds

		self.influence(train, dev, test, classifier, None)
		fds
		# classifier=torch.load('Models/test.pt')
		# fds

		# num_to_add=self.argdict['split']*self.argdict['dataset_size']/len(self.argdict['categories'])
		#
		# self.augmentator.argdict['split']=self.argdict['split']*2
		# aug=self.augmentator.augment(train, return_dict=True)
		# aug=pd.DataFrame.from_dict(aug, orient='index')
		# print(aug)
		# confidence=classifier.get_logits(list(aug['sentence']))
		# confidence=torch.max(torch.softmax(confidence, dim=1), dim=1)[0]
		# print(confidence)
		# asdf

		liste_sent = list(train.return_pandas()['sentence'])
		num_to_add_per_class = self.argdict['dataset_size'] * self.argdict['split'] / len(self.argdict['categories'])
		num_added_per_class = [0] * len(self.argdict['categories'])
		dict_final = {}
		while not checkCondition(num_added_per_class, num_to_add_per_class):
			augmented_exes = self.augmentator.augment(train, return_dict=True)
			aug = pd.DataFrame.from_dict(augmented_exes, orient='index')
			confidence = classifier.get_logits(list(aug['sentence']))
			confidence = torch.argmax(torch.softmax(confidence, dim=1), dim=1)
			for (i, augmented_ex), conf in zip(augmented_exes.items(), confidence):
				if conf==1 and num_added_per_class[augmented_ex['label']] < num_to_add_per_class:
					dict_final[len(dict_final)] = augmented_ex
					liste_sent.append(augmented_ex['sentence'])
					num_added_per_class[augmented_ex['label']] += 1
		#Filtering by confidence
		new_dico_filtered={}
		# for i, ex in aug.items():
		# 	confidence=classifier.

		for j, item in dict_final.items():
			len_data = len(train)
			# print(item)
			train.data[len_data] = item

		# print(len(train))
		# fds

		return train




