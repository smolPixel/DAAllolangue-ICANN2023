"""Wrapper for the SSVAE"""
import os
import json
import time
import torch
import argparse
import shutil
import numpy as np
from multiprocessing import cpu_count
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score

# from Generators.VAE.ptb import PTB
from AugmentStrat.PAE_strat.utils import to_var,expierment_name
from AugmentStrat.PAE_strat.model import SentenceAE
from data.DataProcessor import initialize_dataset_pretraining


class PAE_meta():

    def __init__(self, argdict, train):
        self.argdict=argdict
        self.train=train
        self.language = self.train.language
        self.model, self.params=self.init_model_dataset()
        # optimizers
          # self.argdict.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function_discriminator = torch.nn.CrossEntropyLoss()
        self.step=0
        self.epoch = 0


    def init_model_dataset(self):
        splits = ['train', 'dev']  # + (['test'] if self.argdict.test else [])

        # datasets = OrderedDict()
        # for split in splits:
        #     datasets[split] = PTB(
        #         data_dir=self.argdict['pathFolder'] + '/Generators/VAE/data',
        #         split=split,
        #         create_data=False,  # self.argdict.create_data,
        #         max_sequence_length=60,  # self.argdict.max_sequence_length,
        #         min_occ=0  # self.argdict.min_occ
        #     )

        # print("BLIBLBILBi")
        # print(datasetsLabelled['train'])
        # print(self.train.vocab_size)
        # print(self.train.vocab_size)
        # fds

        self.pretrain_dataset = initialize_dataset_pretraining(self.argdict, self.language)

        params = dict(
            vocab_size=self.pretrain_dataset.vocab_size,
            sos_idx=self.pretrain_dataset.bos_idx,
            eos_idx=self.pretrain_dataset.eos_idx,
            pad_idx=self.pretrain_dataset.pad_idx,
            unk_idx=self.pretrain_dataset.unk_idx,
            max_sequence_length=self.train.max_len,
            embedding_size=300,  # self.argdict.embedding_size,
            rnn_type='gru',  # self.argdict.rnn_type,
            hidden_size=self.argdict['hidden_size_algo'],
            word_dropout=self.argdict['word_dropout'],  # self.argdict.word_dropout,
            embedding_dropout=self.argdict['dropout_algo'],  # self.argdict.embedding_dropout,
            latent_size=self.argdict['latent_size'],
            num_layers=1,  # self.argdict['num_layers_algo'],
            bidirectional=False  # self.argdict.bidirectional
        )


        try:
            model = torch.load(f"/data/rali6/Tmp/piedboef/Models/DACon/PAE_{self.language}_{self.argdict['strat_pretraining']}.pt")
            self.pretraining = False
        except:
            self.model = SentenceAE(**params).cuda()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
            self.pretrain_dataset=initialize_dataset_pretraining(self.argdict, self.language)
            self.pretrain_model()
            torch.save(self.model, f"/data/rali6/Tmp/piedboef/Models/DACon/PAE_{self.language}_{self.argdict['strat_pretraining']}.pt")
            self.pretraining = False
            model=self.model
        # model = SentenceVAE(**params)
        if torch.cuda.is_available():
            model = model.cuda()

        if self.argdict['embeddings'].lower()=="glove":
            from torchtext.vocab import GloVe
            pretrained_vectors=GloVe(name='6B', cache='/data/rali5/Tmp/piedboef/embeddings/.vector_cache/', dim=300)
            model.embedding.weight.data=pretrained_vectors.get_vecs_by_tokens(self.train.vocab.get_itos()).cuda()
            print(f"loaded glove vectors of shape {model.embedding.weight.data.shape}")
        if self.argdict['freeze_embeddings']:
            model.embedding.weight.requires_grad = False

        self.step=0
        self.epoch=0

        return model, params


    def loss_fn(self, logp, target):
        NLL = torch.nn.NLLLoss(ignore_index=self.train.pad_idx, reduction='sum')
        # cut-off unnecessary padding from target, and flatten
        target = target.contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        return NLL_loss


    def run_batches(self, batch, iteration, data_loader, verbose=False):
        batch_size = batch['input'].size(0)
        # print(batch['input'])
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        # Forward pass
        input = batch['input'][:, :-1]
        logp, mean, logv, z = self.model(input)
        # print(batch['input'].shape)
        target = batch['input'][:, 1:]
        # print(target.shape)
        # loss calculation
        # NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
        #                                        batch['length'], mean, logv, self.argdict.anneal_function, step,
        #                                        self.argdict.k, self.argdict.x0)
        NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target)

        loss = (NLL_loss + KL_weight * KL_loss) / batch_size

        # backward + optimization
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.step += 1

        if iteration % 50 == 0 and verbose or iteration + 1 == len(data_loader) and verbose:
            print("Train Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                  % (
                      iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
                      KL_loss.item() / batch_size, KL_weight))
        return NLL_loss.item() / batch_size, KL_loss.item() / batch_size, KL_weight

    def run_epoch(self, verbose=False):
        data_loader = DataLoader(
            dataset=self.train,
            batch_size=32,  # self.argdict.batch_size,
            shuffle= True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        # tracker = defaultdict(tensor)

        # Enable/Disable Dropout
        self.model.train()
        NLL=[]
        KL=[]
        KL_weight=[]

        for iteration, batch in enumerate(data_loader):
            nll_batch, KL_batch, weight_batch= self.run_batches(batch, iteration, data_loader)
            NLL.append(nll_batch)
            KL.append(KL_batch)
            KL_weight.append(weight_batch)
        print(f"Epoch {self.epoch} KL div {np.meant(KL)} KL Weight {np.mean(KL_weight)}, NLL {np.mean(NLL)}")


    def pretrain_model(self):
        for epoch in range(self.argdict['nb_epoch_algo_pretraining']):
            data_loader = DataLoader(
                dataset=self.pretrain_dataset,
                batch_size=32,  # self.argdict.batch_size,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            self.model.train()
            NLL = []
            KL = []
            KL_weights = []

            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].size(0)
                # print(batch['input'])
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                input = batch['input'][:, :-1]

                logp = self.model(input, pretraining=True)
                # print(batch['input'].shape)
                target = batch['input'][:, 1:]
                # print(target.shape)
                # loss calculation
                # NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                #                                        batch['length'], mean, logv, self.argdict.anneal_function, step,
                #                                        self.argdict.k, self.argdict.x0)
                # NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target, mean, logv, 'logistic', 0,
                #                                             0.0025, pretraining=True)


                NLL_perte= torch.nn.NLLLoss(ignore_index=self.pretrain_dataset.pad_idx)
                #(NLL_loss + KL_weight * KL_loss) / batch_size
                target = target.contiguous().view(-1)
                logp = logp.view(-1, logp.size(2))

                # Negative Log Likelihood
                NLL_loss = NLL_perte(logp, target)
                loss=NLL_loss
                # backward + optimization
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
                self.optimizer.step()
                NLL.append(NLL_loss.cpu().detach())
                # KL.append(KL_loss)
                # KL_weights.append(KL_weight)
            print(f"Epoch {epoch} KL div {np.mean(KL)} KL Weight {np.mean(KL_weights)}, NLL {np.mean(NLL)}")
            torch.save(self.model, f"/data/rali6/Tmp/piedboef/Models/DACon/PVAE_{self.language}_{self.argdict['strat_pretraining']}_Ep{epoch}.pt")

    def train_model(self, verbose=False):
        save_model_path = os.path.join(self.argdict['path'], 'bin')
        # shutil.
        os.makedirs(save_model_path, exist_ok=True)

        # with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        #     json.dump(self.params, f, indent=4)



        # tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        # step = 0
        # for epoch in range(self.argdict.epochs):
        for epoch in range(self.argdict['nb_epoch_algo']):
            self.epoch=epoch
            self.run_epoch()
            # self.generate_from_train()
            self.interpolate(5)
        fds

                # if self.argdict.tensorboard_logging:
                #     writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

                # save a dump of all sentences and the encoded latent space
                # if split == 'valid':
                #     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                #     if not os.path.exists(os.path.join('dumps', ts)):
                #         os.makedirs('dumps/' + ts)
                #     with open(os.path.join('dumps/' + ts + '/valid_E%i.json' % epoch), 'w') as dump_file:
                #         json.dump(dump, dump_file)

                # save checkpoint
                # if split == 'train':
                #     checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                #     torch.save(self.model.state_dict(), checkpoint_path)
                #     print("Model saved at %s" % checkpoint_path)

    def generate_from_train(self, n=2):
        #Generate x data points from train
        data_loader = DataLoader(
            dataset=self.train,
            batch_size=n,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        # tracker = defaultdict(tensor)

        # Enable/Disable Dropout
        self.model.eval()

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            z = self.model.encode(batch['input'])
            generated, z = self.model.inference(n=n, z=z)
            generated = self.train.arr_to_sentences(generated)

            print(f"input: {self.train.arr_to_sentences(batch['input'])}\n"
                  f"Note marginale: {batch['NoteMarginale']}\n"
                  f"Words: {self.train.arr_to_sentences(batch['words'])}\n"
                  f"Generated: {generated}")
            break

    def interpolate(self, n=5):
        p0=to_var(torch.randn([1, self.argdict['latent_size']]))
        p1=to_var(torch.randn([1, self.argdict['latent_size']]))
        points=torch.zeros(n, self.argdict['latent_size'])
        points[0]=p0
        points[n-1]=p1
        for i in range(n):
            ratio=i/n
            px=(1-ratio)*p0+ratio*p1
            # if i<(n/2):
            #     points[i]=p0
            # else:
            #     points[i]=p1
            points[i]=px
        points=points.cuda()
        # print(points)
        samples, z = self.model.inference(n=n, z=points)
        generated = self.train.arr_to_sentences(samples)
        print("Interpolation:")
        for sent, latent in zip(generated, z):
            print("------------------")
            # print(latent)
            print(sent)
        # fsd

    def encode_examples(self, input):
        z=self.model.encode(input)
        return z


    def encode(self):
        dico={}
        data_loader = DataLoader(
            dataset=self.train,
            batch_size=64,#self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        # Enable/Disable Dropout

        self.model.eval()
        # print(f"The dataset length is {len(data_loader.dataset)}")
        dataset = torch.zeros(len(data_loader.dataset), self.params['latent_size'])
        labels = torch.zeros(len(data_loader.dataset))
        counter = 0
        for iteration, batch in enumerate(data_loader):
            # print("Oh la la banana")
            batch_size = batch['input'].size(0)
            # print(batch['input'].shape)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)
            #
            # print(batch['input'])
            # print(batch['input'].shape)
            z = self.model.encode(batch['input'])
            # print(batch_size)
            # print(z.shape)
            dataset[counter:counter + batch_size] = z
            labels[counter:counter + batch_size] = batch['label']
            counter += batch_size
        # print(dataset)
        dico[f"labels"]=labels
        dico[f"encoded"]=dataset
        # torch.save(labels, f"bin/labels_{split}.pt")
        # torch.save(dataset, f"bin/encoded_{split}.pt")
        return dico

    def generate(self, datapoints=None):
        #Generates from fixed datapoints
        self.model.eval()

        samples, z = self.model.inference(z=datapoints)
        # print(samples)
        # print('----------SAMPLES----------')
        return self.train.arr_to_sentences(samples)

