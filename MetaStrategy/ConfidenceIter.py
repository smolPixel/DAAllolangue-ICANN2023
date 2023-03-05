import pandas as pd
import torch
import copy

def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

class confidence_iter_metastrat():
	"""Semi supervised learning with augmented data"""
	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.argdict=argdict
		self.augmentator=augmentator
		self.pool=10000 #Generate 10000 examples at each iterations
		self.top_k=100 #Takes the 100 best at each iteration

	def prep_algo(self, train):
		from Classifier.classifier import classifier
		self.classifier = classifier(argdict=self.argdict)
		self.classifier.train_test(train, train)


	def evaluate(self, augmented_exes, train):
		Verdicts = []
		aug = pd.DataFrame.from_dict(augmented_exes, orient='index')
		confidence = self.classifier.get_logits(list(aug['sentence']))
		confidence = torch.max(torch.softmax(confidence, dim=1), dim=1)[0]
		for (i, augmented_ex), conf in zip(augmented_exes.items(), confidence):
			if conf > 0.75:
				Verdicts.append(True)
			else:
				Verdicts.append(False)
		return Verdicts

	def augment(self, train):
		# self.prep_algo(train)
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
		trainTemp=copy.deepcopy(train)
		liste_sent = list(train.return_pandas()['sentence'])
		num_to_add_per_class = self.argdict['dataset_size'] * self.argdict['split'] / len(self.argdict['categories'])
		num_added_per_class = [0] * len(self.argdict['categories'])
		dict_final = {}
		while not checkCondition(num_added_per_class, num_to_add_per_class):
			#For each iteration. First train the
			self.prep_algo(trainTemp)

			#TODO change here so that it returns 10K examples instead
			augmented_exes = self.augmentator.augment(train, return_dict=True)
			aug = pd.DataFrame.from_dict(augmented_exes, orient='index')
			confidence = self.classifier.get_logits(list(aug['sentence']))
			confidence = torch.max(torch.softmax(confidence, dim=1), dim=1)[0]
			conf, indexes=torch.topk(confidence, self.top_k)
			for cc, ii in zip(conf, indexes):
				augmented_ex = augmented_exes[ii.cpu().item()]
				if num_added_per_class[augmented_ex['label']] < num_to_add_per_class:
					dict_final[len(dict_final)] = augmented_ex
					liste_sent.append(augmented_ex['sentence'])
					num_added_per_class[augmented_ex['label']] += 1
					trainTemp[len(trainTemp)]=augmented_ex
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




