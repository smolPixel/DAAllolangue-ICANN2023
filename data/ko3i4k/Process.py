import pandas as pd

df=pd.DataFrame(columns=["sentence", "label"])

# file=open('fci_train_val.txt', 'r')
#
# for i, line in enumerate(file.readlines()):
# 	label, text= line.split("\t")
# 	df.at[i, 'sentence']=text
# 	df.at[i, 'label']=label
#
# dev=df.sample(frac=0.05, axis=0)
# train=df.drop(index=dev.index)
# train=train.reset_index()
# dev=dev.reset_index()
#
#
# train.to_csv('train.tsv', sep='\t')
# dev.to_csv('dev.tsv', sep='\t')

file=open('fci_test.txt', 'r')

for i, line in enumerate(file.readlines()):
	label, text= line.split("\t")
	df.at[i, 'sentence']=text.strip()
	df.at[i, 'label']=label

print(df)
df.to_csv('test.tsv', sep='\t')
