"""Checking for bleeding data"""

import pandas as pd

train=list(pd.read_csv('train.tsv', sep='\t', index_col=0)['sentence'])
dev=list(pd.read_csv('dev.tsv', sep='\t', index_col=0)['sentence'])
test=list(pd.read_csv('test.tsv', sep='\t', index_col=0)['sentence'])

bleed_trainDev=0
bleed_trainTest=0

for ss in train:
	if ss in dev:
		bleed_trainDev+=1
	if ss in test:
		bleed_trainTest+=1
		print(ss)

print(bleed_trainDev)
print(bleed_trainTest)
