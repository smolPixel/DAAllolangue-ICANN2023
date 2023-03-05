import pandas as pd
import torch
def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

class gradient_metastrat():

	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.argdict=argdict
		self.augmentator=augmentator

	def augment(self, train):
		from Classifier.classifier import classifier
		classifier=classifier(argdict=self.argdict)
		classifier.train_test(train, train)

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
			# print(list(aug))
			gradients = classifier.get_grads(list(aug['sentence']), list(aug['label']))
			# print(gradients)
			# fds
			# confidence = torch.max(torch.softmax(confidence, dim=1), dim=1)[0]
			for (i, augmented_ex), grad in zip(augmented_exes.items(), gradients):
				if grad>10 and num_added_per_class[augmented_ex['label']] < num_to_add_per_class:
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




