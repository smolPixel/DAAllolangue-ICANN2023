from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from process_data import *
import torch
import os
from sklearn.metrics import accuracy_score



class Bert_Labeller():

	def __init__(self, argdict):
		self.argdict=argdict

		# print(self.model)


	# def run_epoch(self, train, model, tokenizer, optimizer):
	#
	#     data_loader = DataLoader(
	#         dataset=train,
	#         batch_size=64,
	#         shuffle=True,
	#         # num_workers=cpu_count(),
	#         pin_memory=torch.cuda.is_available()
	#     )
	#
	#     for i, batch in enumerate(data_loader):
	#         optimizer.zero_grad()
	#         # print(batch)
	#         # fds
	#         text_batch=batch['sentence']
	#         encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
	#         input_ids = encoding['input_ids'].cuda()
	#         attention_mask = encoding['attention_mask'].cuda()
	#         # print(encoding)
	#         labels=batch['true_label'].cuda()
	#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	#         # print(outputs)
	#         # print(loss, outputs)
	#         # print(outputs)
	#         loss = outputs.loss
	#         loss.backward()
	#         optimizer.step()
	#
	#     #Test
	#     return 0

	# def calculateAccuracyDev(self, dev, model, tokenizer):
	#     data_loader = DataLoader(
	#         dataset=dev,
	#         batch_size=64,
	#         shuffle=False,
	#         # num_workers=cpu_count(),
	#         pin_memory=torch.cuda.is_available()
	#     )
	#     pred = torch.zeros(len(dev))
	#     Y = torch.zeros(len(dev))
	#     start = 0
	#     for i, batch in enumerate(data_loader):
	#         with torch.no_grad():
	#             text_batch = batch['sentence']
	#             encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
	#             input_ids = encoding['input_ids'].cuda()
	#             attention_mask = encoding['attention_mask'].cuda()
	#             # print(encoding)
	#             labels = batch['label'].cuda()
	#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	#             results = torch.argmax(torch.log_softmax(outputs.logits, dim=1), dim=1)
	#             pred[start:start + 64] = results
	#             Y[start:start + 64] = batch['label']
	#             start = start + 64
	#     return accuracy_score(Y, pred)

	def train(self):
		# self.tokenizer = BertTokenizer.from_pretrained('Models/bert_labellers_tokenizer.ptf')
		self.model = BertForSequenceClassification.from_pretrained('AugmentStrat/transformersintovaes/Models/bert_labeller').cuda()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# acc = self.calculateAccuracyDev(dev, self.model, self.tokenizer)
		print("Model has already been trained")

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