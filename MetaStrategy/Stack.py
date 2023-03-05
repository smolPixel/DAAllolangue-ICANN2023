"""Filter examples by removing doublons"""
from MetaStrategy.MetaStrat import meta_strat
from MetaStrategy.Doublons import doublons_metastrat
import copy

def checkCondition(array, num):
	# filled=True
	for numAdded in array:
		if numAdded<num:
			return False
	return True

class stack_metastrat():

	def __init__(self, argdict, augmentator):
		"""Does not change the augmentator"""
		self.augmentator=augmentator
		self.argdict=argdict
		# self.strats=['doublons', 'confidence']
		self.strats=['confidence']
		self.strats_algo=[]
		argdicts=[copy.deepcopy(argdict) for i in range(len(self.strats))]
		for i, strat in enumerate(self.strats):
			argdicts[i]['meta_strategy']=self.strats[i]
			self.strats_algo.append(meta_strat(argdicts[i], augmentator))

	def augment(self, train):
		for strat in self.strats_algo:
			strat.prep_algo(train)
		liste_sent=list(train.return_pandas()['sentence'])
		num_to_add_per_class=self.argdict['dataset_size']*self.argdict['split']/len(self.argdict['categories'])
		num_added_per_class=[0]*len(self.argdict['categories'])
		dict_final={}
		while not checkCondition(num_added_per_class, num_to_add_per_class):
			augmented_exes=self.augmentator.augment(train, return_dict=True)
			Verdicts = [strat.evaluate(augmented_exes, train) for strat in self.strats_algo]
			for i, augmented_ex in augmented_exes.items():
				verdict_example=[Verdicts[j][i] for j in range(len(self.strats))]
				if False not in verdict_example and num_added_per_class[augmented_ex['label']]<num_to_add_per_class:
					dict_final[len(dict_final)]=augmented_ex
					liste_sent.append(augmented_ex['sentence'])
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



