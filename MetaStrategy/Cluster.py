import pandas as pd
import torch
import sys
from scipy.spatial.distance import cosine

def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

"""Cluster metastrat: In the train algorithm, calculate the average position of each class. If a data point is further
from its supposed class average than from the median from any other class, reject it"""
class cluster_metastrat():

	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.argdict=argdict
		self.augmentator=augmentator

	def augment(self, train):
		from Classifier.classifier import classifier
		args=self.argdict
		args['output_hidden_state']=True
		classifier=classifier(argdict=args)
		classifier.train_test(train, train)
		with torch.no_grad():


			# print(train.return_pandas())
			pdTrain=train.return_pandas()
			sentsPos=[sent for sent, lab in zip(list(pdTrain['sentence']), list(pdTrain['label'])) if lab==1]
			sentsNeg=[sent for sent, lab in zip(list(pdTrain['sentence']), list(pdTrain['label'])) if lab==0]
			# print(len(sentsPos))
			repPos=classifier.get_rep(sentsPos)
			repNeg=classifier.get_rep(sentsNeg)

		repPos=torch.mean(repPos, dim=0)
		repNeg=torch.mean(repNeg, dim=0)
		reps=[repNeg.cpu(), repPos.cpu()]

		liste_sent = list(train.return_pandas()['sentence'])
		num_to_add_per_class = self.argdict['dataset_size'] * self.argdict['split'] / len(self.argdict['categories'])
		num_added_per_class = [0] * len(self.argdict['categories'])
		dict_final = {}
		iter=0
		while not checkCondition(num_added_per_class, num_to_add_per_class):
			sys.stdout.write('\r')
			percent = int(sum(num_added_per_class) * 100 / (num_to_add_per_class * len(self.argdict['categories'])))
			sys.stdout.write("[%-100s] %d%%, Iteration n.%d" % ('=' * percent, percent, iter))
			iter += 1
			if iter == 40:
				raise OverflowError("Cutoff at 40 iterations")
			sys.stdout.flush()


			augmented_exes = self.augmentator.augment(train, return_dict=True)
			aug = pd.DataFrame.from_dict(augmented_exes, orient='index')
			with torch.no_grad():
				position = classifier.get_rep(list(aug['sentence']))
			labels=list(aug['label'])

			for (i, augmented_ex), pos in zip(augmented_exes.items(), position):
				# print(augmented_ex, pos)
				dists=[cosine(pos.cpu(), reps[i]) for i in range(2)]
				# print(dists)
				label=augmented_ex['label']
				if max(dists)==dists[label] and num_added_per_class[augmented_ex['label']] < num_to_add_per_class:
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




