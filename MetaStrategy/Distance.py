import pandas as pd
import torch
# from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from SentAugment.src.sase import sase
from SentAugment.src.compress_text import compress_text
from SentAugment.src.flat_retrieve import flat_retrieve
from tqdm import tqdm
import sys


import subprocess
def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

def run_external_process(process):
	output, error = process.communicate()
	if process.returncode != 0:
		raise SystemError
	return output, error
class distance_metastrat():

	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.argdict=argdict
		self.augmentator=augmentator
		self.vectorizer=CountVectorizer(ngram_range=(1,1))

	def compute_graph(self, train):
		from Classifier.classifier import classifier
		self.argdict['output_hidden_state']=True
		classifier = classifier(argdict=self.argdict)
		classifier.train_test(train, train)

		augmented_exes = self.augmentator.augment(train, return_dict=True)
		aug = pd.DataFrame.from_dict(augmented_exes, orient='index')
		og=train.return_pandas()
		sentences_og=list(og['sentence'])
		sentences_aug=list(aug['sentence'])
		labels_og = list(og['label'])
		labels_aug = list(aug['label'])
		"""Distance in BERT"""
		rep_og=classifier.get_rep(sentences_og)
		rep_aug=classifier.get_rep(sentences_aug)

		all_vect=torch.cat([rep_og, rep_aug], dim=0).cpu()
		matrix = cosine_similarity(all_vect)[1000:, :1000]
		# Each line is an augmented example, each column is an og datapoint
		# print(matrix)
		# Closest sentence
		dist = np.max(matrix, axis=1)
		doc = np.argmax(matrix, axis=1)
		# print(dist)
		wrong_labels = []
		right_labels = []
		for i, (document, distance) in enumerate(zip(doc, dist)):
			# print(i, document, distance)
			# Get label of both the og and augmented example:
			lab_aug = labels_aug[i]
			lab_og = labels_og[document]
			# print(lab_aug, lab_og)
			# print(sentences_aug[i])
			# print(sentences_og[document])
			if lab_aug != lab_og:
				wrong_labels.append(distance)
			else:
				right_labels.append(distance)
		plt.figure()
		plt.hist(right_labels, bins=50, label='Same label')
		plt.hist(wrong_labels, bins=50, label='Wrong label')
		plt.legend()
		# plt.xticks(splits)
		plt.xlabel("BERT distance to the closest example")
		plt.ylabel("Frequency")

		plt.savefig("Graphes/Bert_CLS_cosine_Dist.png")
		fds

		"""Distance SASE"""
		file = "\n".join(sentences_og)
		with open("transformed_og.txt", "w") as ff:
			ff.write(file)

		bashCommand = 'python3 SentAugment/src/sase.py --input transformed_og.txt --model data/sase.pth ' \
					  '--spm_model data/sase.spm --batch_size 64 --cuda True --output transformed_og.pt'
		process = subprocess.Popen(bashCommand.split())
		output, error = run_external_process(process)
		embeds_og = torch.load("transformed_og.pt")
		# fds
		#
		# all_vect=torch.cat([embeds_og, embeds_aug], dim=0)

		# Create bank
		bashCommand = 'python3 SentAugment/src/compress_text.py --input transformed_og.txt'
		process = subprocess.Popen(bashCommand.split())
		output, error = run_external_process(process)
		file = "\n".join(sentences_aug)
		with open("transformed_aug.txt", "w") as ff:
			ff.write(file)

		bashCommand = 'python3 SentAugment/src/sase.py --input transformed_aug.txt --model data/sase.pth ' \
					  '--spm_model data/sase.spm --batch_size 64 --cuda True --output transformed_aug.pt'
		process = subprocess.Popen(bashCommand.split())
		output, error = run_external_process(process)
		embeds_aug = torch.load("transformed_aug.pt")
		from SentAugment.src.flat_retrieve import flat_retrieve
		# bashCommand='python3 SentAugment/src/flat_retrieve.py --input transformed_aug.pt --bank transformed_og.txt --emb transformed_og.pt --K 1'
		# with open(f"temp.txt", "w") as outfile:
		# 	process = subprocess.Popen(bashCommand.split(), stdout=outfile)
		# 	output, error = run_external_process(process)
		distances, closest_sentences=flat_retrieve('transformed_aug.pt', 'transformed_og.txt', 'transformed_og.pt', 1)
		labels_og=list(og['label'])
		labels_aug=list(aug['label'])

		true_labels_ref=[]
		for og_sentence, closest_sentence in zip(sentences_og, closest_sentences):
			ind=sentences_og.index(closest_sentence)
			true_labels_ref.append(labels_og[ind])

		distances=distances.squeeze(0)
		# print(distances)
		# fds
		wrong_labels = []
		right_labels = []
		for i, dist in enumerate(distances):
			# print(i, document, distance)
			#Get label of both the og and augmented example:
			lab_aug=labels_aug[i]
			lab_og=true_labels_ref[i]
			# print(lab_aug, lab_og)
			# print(sentences_aug[i])
			# print(sentences_og[document])
			# print(dist)
			# print(dist.item())
			if lab_aug!=lab_og:
				wrong_labels.append(dist.item())
			else:
				right_labels.append(dist.item())
		plt.figure()
		plt.hist(right_labels, bins=50, label='Same label')
		plt.hist(wrong_labels, bins=50, label='Wrong label')
		plt.legend()
		# plt.xticks(splits)
		plt.xlabel("SASE distance to the closest example")
		plt.ylabel("Frequency")

		plt.savefig("Graphes/sase_Dist.png")
		fds
		# print(labels_og)
		# fds
		# sentences_aug[1]=sentences_og[0]
		# all_sentences=sentences_og#.extend(sentences_aug)
		# all_sentences.extend(sentences_aug)
		# print(all_sentences)
		# all_vect=self.vectorizer.fit_transform(all_sentences)
		# og_vect=self.vectorizer.transform(sentences_og)
		# aug_vect=self.vectorizer.transform(sentences_aug)
		# print(cosine_similarity(all_vect).shape)
		#1000 first are the og, 1000 last are the augmented, so the matrix og aug is
		# matrix=cosine_similarity(all_vect)[1000: , :1000]
		#Each line is an augmented example, each column is an og datapoint
		# print(matrix)
		#Closest sentence
		dist=np.max(matrix, axis=1)
		doc=np.argmax(matrix, axis=1)
		# print(dist)
		wrong_labels=[]
		right_labels=[]
		for i, (document, distance) in enumerate(zip(doc, dist)):
			# print(i, document, distance)
			#Get label of both the og and augmented example:
			lab_aug=labels_aug[i]
			lab_og=labels_og[document]
			# print(lab_aug, lab_og)
			# print(sentences_aug[i])
			# print(sentences_og[document])
			if lab_aug!=lab_og:
				wrong_labels.append(distance)
			else:
				right_labels.append(distance)
		plt.figure()
		plt.hist(right_labels, bins=50, label='Same label')
		plt.hist(wrong_labels, bins=50, label='Wrong label')
		plt.legend()
		# plt.xticks(splits)
		plt.xlabel("TF-IDF distance to the closest example")
		plt.ylabel("Frequency")

		plt.savefig("Graphes/sase_cosine_Dist.png")
		fds

	def compute_bert_distance(self, og_sentences, aug_sentences):
		pass

	def compute_sase_distance(self, og_sentences, aug_sentences):
		embeds_og=sase(og_sentences, 'data/sase.pth', 'data/sase.spm', batch_size=64, cuda=True)
		compress_text(og_sentences)
		embeds_aug=sase(aug_sentences, 'data/sase.pth', 'data/sase.spm', batch_size=64, cuda=True)
		distances, closest_sentences = flat_retrieve(embeds_aug, og_sentences, embeds_og, 1)
		distances = distances.squeeze(0)
		return distances

	def compute_tf_distance(self, og_sentences, aug_sentences):
		# sentences_aug[1]=sentences_og[0]
		all_sentences=og_sentences#.extend(sentences_aug)
		all_sentences.extend(aug_sentences)
		# print(all_sentences)
		all_vect=self.vectorizer.fit_transform(all_sentences)
		# og_vect=self.vectorizer.transform(sentences_og)
		# aug_vect=self.vectorizer.transform(sentences_aug)
		# print(cosine_similarity(all_vect).shape)
		# 1000 first are the og, 1000 last are the augmented, so the matrix og aug is
		matrix=cosine_similarity(all_vect)[1000: , :1000]
		# Each line is an augmented example, each column is an og datapoint
		# print(matrix)
		# Closest sentence
		dist = np.max(matrix, axis=1)
		doc = np.argmax(matrix, axis=1)

		return dist

	def augment(self, train):
		# self.compute_graph(train)


		liste_sent = list(train.return_pandas()['sentence'])
		num_to_add_per_class = self.argdict['dataset_size'] * self.argdict['split'] / len(self.argdict['categories'])
		num_added_per_class = [0] * len(self.argdict['categories'])
		dict_final = {}
		iter=0
		while not checkCondition(num_added_per_class, num_to_add_per_class):
			#Progress bar
			sys.stdout.write('\r')
			percent=int(sum(num_added_per_class)*100/(num_to_add_per_class*len(self.argdict['categories'])))
			sys.stdout.write("[%-100s] %d%%, Iteration n.%d" % ('=' * percent, percent, iter))
			iter+=1
			if iter==40:
				raise OverflowError("Cutoff at 40 iterations")
			sys.stdout.flush()
			augmented_exes = self.augmentator.augment(train, return_dict=True)
			aug = pd.DataFrame.from_dict(augmented_exes, orient='index')
			distances=self.compute_sase_distance(liste_sent, aug['sentence'])
			for (i, augmented_ex), distance in zip(augmented_exes.items(), distances):
				if distance<=0.9 and num_added_per_class[augmented_ex['label']] < num_to_add_per_class:
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




