"""Filter examples by removing doublons"""

def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

class doublons_metastrat():

	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.augmentator=augmentator
		self.argdict=argdict

	def prep_algo(self, train):
		self.liste_sent = list(train.return_pandas()['sentence'])


	def evaluate(self, augmented_exes, train):
		Verdicts=[]
		for i, augmented_ex in augmented_exes.items():
			if augmented_ex['sentence'] not in self.liste_sent:
				self.liste_sent.append(augmented_ex['sentence'])
				Verdicts.append(True)
			else:
				Verdicts.append(False)
		return Verdicts


	def augment(self, train):
		self.prep_algo(train)
		num_to_add_per_class=self.argdict['dataset_size']*self.argdict['split']/len(self.argdict['categories'])
		num_added_per_class=[0]*len(self.argdict['categories'])
		dict_final={}
		while not checkCondition(num_added_per_class, num_to_add_per_class):
			augmented_exes=self.augmentator.augment(train, return_dict=True)
			for i, augmented_ex in augmented_exes.items():
				if augmented_ex['sentence'] not in self.liste_sent and num_added_per_class[augmented_ex['label']]<num_to_add_per_class:
					dict_final[len(dict_final)]=augmented_ex
					self.liste_sent.append(augmented_ex['sentence'])
					num_added_per_class[augmented_ex['label']]+=1
			# print(augmented_ex)
		# print(len(dict_final))
		# print(len(train))
		# fds
		for j, item in dict_final.items():
			len_data = len(train)
			# print(item)
			train.data[len_data] = item

		# print(len(train))
		# fds

		return train



