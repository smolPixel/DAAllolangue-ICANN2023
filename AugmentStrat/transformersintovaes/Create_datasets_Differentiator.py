import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.DataFrame(columns=['sentence', 'label'])

f=open('Generated_Bart_No_FineTune.txt', 'r').readlines()

index=0
for line in f:
	df.at[index, 'sentence']=line.replace('\n', ' ')
	df.at[index, 'label']=0
	index+=1


df_SST=pd.read_csv("train_SST.tsv", sep='\t')
for _, row in df_SST.iterrows():
	df.at[index, 'sentence']=row['sentence']
	df.at[index, 'label']=1
	index+=1


train, dev=train_test_split(df, test_size=0.2, shuffle=True)

# print(df)

train.to_csv('train_differentiator.tsv', sep='\t')
dev.to_csv('dev_differentiator.tsv', sep='\t')