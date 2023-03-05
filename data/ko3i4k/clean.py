import pandas as pd

name='dev.tsv'

df=pd.read_csv(name, sep='\t', index_col=0)
df=df.drop(columns='index')
for i, row in df.iterrows():
	df.at[i, 'sentence']=row['sentence'].strip()

df.to_csv(name, sep='\t')