import pandas as pd

name='test.tsv'

df=pd.read_csv(f'OG/{name}', sep='\t', index_col=0)
for i, row in df.iterrows():
	ss=row['sentence'].split('.')[0]
	df.at[i, 'sentence']=ss

df.to_csv(name, sep='\t')