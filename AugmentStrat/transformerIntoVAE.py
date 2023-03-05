"""VAE augmentation"""
import pandas
from AugmentStrat.VAE_strat.VAE import VAE_meta
from data.DataProcessor import separate_per_class
import torch
from AugmentStrat.VAE_strat.utils import to_var
import math
import itertools
import numpy as np
import random
import subprocess
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from data.DataProcessor import separate_per_class
import shutil

def run_external_process(process):
	output, error = process.communicate()
	if process.returncode != 0:
		raise SystemError
	return output, error


class VAE():

	def __init__(self, argdict):
		self.argdict=argdict
		self.algo_is_trained=False
		self.vae_per_class=[None]*len(self.argdict['categories'])


	def sample_latent_space(self, bs, encoded):
		if self.argdict['sampling_strategy']=="random":
			return to_var(torch.randn([bs, self.argdict['latent_size']]))
		elif self.argdict['sampling_strategy']=="grid":
			root=math.ceil(bs**(1/self.argdict['latent_size']))
			#Spans from -1 to 1, we need root spaces
			largeur_col=3/float((root-1))

			dim = [-1.5 + largeur_col * i for i in range(root)]
			all_possibilities = []
			for comb in itertools.product(dim, repeat=self.argdict['latent_size']):
				all_possibilities.append(comb)
			point=torch.zeros(bs, self.argdict['latent_size'])
			points_chosen=np.random.choice(np.arange(len(all_possibilities)), size=bs, replace=False)
			points_latent=torch.zeros([bs, self.argdict['latent_size']])
			for i, pp in enumerate(points_chosen):
				comb=all_possibilities[pp]
				points_latent[i]=torch.Tensor(list(comb))
			return points_latent
		elif self.argdict['sampling_strategy']=='posterior':
			enco=encoded['encoded']
			points_latent = torch.zeros([bs, self.argdict['latent_size']])
			num_points=encoded['encoded'].shape[0]
			for i in range(bs):
				random_zero=random.randint(0, num_points-1)
				random_one=random.randint(0, num_points-1)
				interpol=random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
				points_latent[i]=enco[random_zero]*interpol+enco[random_one]*(1-interpol)
			# print(points_latent)
			# pass
			return points_latent

	#Next: train batch pos, batch neg
	def init_models(self, training_sets):
		pass

		# fds

	def train_models(self, train, dev):

		GENERATE_SENT_UNTRAINED=False

		#First, drop data in correct form for this
		sents_train=list(train.return_pandas()['sentence'])
		sents_train="\n".join(sents_train)
		with open("data/optimus/current/train.txt", "w") as f:
			f.write(sents_train)
		sents_dev = list(dev.return_pandas()['sentence'])
		sents_dev = "\n".join(sents_dev)
		with open("data/optimus/current/valid.txt", "w") as f:
			f.write(sents_dev)
		with open("data/optimus/current/test.txt", "w") as f:
			f.write(sents_dev)
		# #Dropout 0.15
		#Load BERT for emotion classification
		fds
		from AugmentStrat.transformersintovaes.fine_tune_normal import finetune_model
		#Phase 1
		KL_thresh=3

		if len(train)==500:
			dataset='Current500'
		if GENERATE_SENT_UNTRAINED:
			model, tokenizer = finetune_model(
				'AugmentStrat/transformersintovaes/pretraining_configs/t5_vae_32_max_none_0.15_yf.yaml',
				dataset,
				True,
				32,
				latent_dim=self.argdict['latent_size'],
				checkpoint=None,
				n_epoch=1,
				KL_thresh=KL_thresh,
				denoising=0.4,
				pooling='max',
				pretrain_encoder=True,
				encoder_only=True,
				return_untrained_model=True)

			sents=[]
			for i in range(10000):
				print(i)
				from AugmentStrat.transformersintovaes.generate import generate
				output = generate(model, [model.config.decoder_start_token_id],
								  sampled_z=torch.zeros((1, self.argdict['latent_size'])).normal_(mean=0, std=1).to(
									  model.device))

				sentAug = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
				sents.append(sentAug)
			with open("Generated_Bart_No_FineTune.txt", "w") as f:
				f.write("\n".join(sents))
		# fds



		# path='/u/piedboef/Documents/DAControlled/lightning_logs/Current500_t5_vae_32_max_none_0.15_yf_bs32/checkpoints/epoch=9-step=159-v6.ckpt'
		model=finetune_model('AugmentStrat/transformersintovaes/pretraining_configs/t5_vae_32_max_none_0.15_yf.yaml',
							dataset,
							True,
							32,
							latent_dim=self.argdict['latent_size'],
							checkpoint=None,
							n_epoch=10,
							KL_thresh=KL_thresh,
							denoising=0.4,
							pooling='max',
							pretrain_encoder=True,
							encoder_only=True)
		#Phase 2
		model=finetune_model('AugmentStrat/transformersintovaes/pretraining_configs/t5_vae_32_max_none_0.15_yf.yaml',
							dataset,
							True,
							32,
							latent_dim=self.argdict['latent_size'],
							checkpoint=model,
							n_epoch=100,
							KL_thresh=KL_thresh,
							denoising=0.4,
							pooling='max',
							pretrain_encoder=False,
							encoder_only=False)

		# fds
		from AugmentStrat.transformersintovaes.evaluate_all import evaluate
		model, tokenizer=evaluate(checkpoint=model, #'/u/piedboef/Documents/DAControlled/lightning_logs/Current500_t5_vae_32_max_none_0.15_yf_bs32/checkpoints/epoch=9-step=159-v6.ckpt',
							config_file="AugmentStrat/transformersintovaes/pretraining_configs/t5_vae_32_max_none_0.15_yf.yaml",
							dataset_name=dataset,
							batch_size=32,
							KL_thresh=KL_thresh,
							denoising=0.4,
							pooling='max',
							latent_dim=self.argdict['latent_size'],
							plot=False)
		self.model, self.tokenizer = model, tokenizer


	def augment(self, train, dev, return_dict=False):
		training_sets = separate_per_class(train)
		dev_sets=separate_per_class(dev)
		diconew = {}
		sents=[]
		labels=[]
		for i, cat in enumerate(self.argdict['categories']):
			self.train_models(training_sets[i], dev_sets[i])
			from AugmentStrat.transformersintovaes.generate import generate

			# Sentences=[ss.split(" ") for ss in Sentences]
			# print(tokenizer(Sentences, return_tensors='pt', padding=True))
			with torch.no_grad():
				# from AugmentStrat.transformersintovaes.generation_utils import GenerationMixin
				#
				# mixin=GenerationMixin(model.t5.decoder)
				# mixin.generate(bos_token_id=3, past_key_values=sample_z)
				#
				num_gen=0
				while num_gen!=500:
					output = generate(self.model, [self.model.config.decoder_start_token_id],
									  sampled_z=torch.zeros((1, self.argdict['latent_size'])).normal_(mean=0, std=1).to(self.model.device))

					sentAug=self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
					if sentAug=="":
						continue
					sents.append(sentAug)
					labels.append(i)
					diconew[len(diconew)] = {'sentence': sentAug,
											 'label': i,
											 'input': train.tokenize(sentAug),
											 'augmented': True}
					num_gen+=1
			# print('------------')
			del self.tokenizer, self.model
		self.algo_is_trained = True
		if return_dict:
			return diconew
		for j, item in diconew.items():
			len_data = len(train)
			# print(item)
			train.data[len_data] = item
		# print(train)
		# print("BTICH")
		# train.return_pandas().to_csv(f'/Tmp/train_test.tsv', sep='\t')
		print("Labelling error")
		from AugmentStrat.transformersintovaes.BERT_Classifier import Bert_Labeller
		bert=Bert_Labeller(self.argdict)
		bert.train()
		pred=bert.label(sents)
		# print(labels)
		from sklearn.metrics import accuracy_score, confusion_matrix
		print(accuracy_score(labels, pred[0]))
		print(confusion_matrix(labels, pred[0]))
		print("Generation accuracy")
		from AugmentStrat.transformersintovaes.BERT_Differentiator import Bert_Differentiator
		bert=Bert_Differentiator()
		bert.train()
		pred=bert.label(sents)
		# print(labels)
		num_ex=len(pred[0])
		labels=[1]*num_ex
		from sklearn.metrics import accuracy_score, confusion_matrix
		print(accuracy_score(labels, pred[0]))


		print(f"Neg: {sents[:2]}")
		print(f"Pos: {sents[-2:]}")
		shutil.rmtree('lightning_logs')
		# fds

		# fds
		return train

		# Creating new dataset
