import pandas as pd

split='test.tsv'

df=pd.read_csv(f"old/{split}", sep='\t', index_col=0)
max_len=20


for i, row in df.iterrows():
	ss=row['sentence'].replace('\n', ' ')
	ss=ss.split('.')
	new_sent=""
	cur_len=0
	for sent in ss:
		if sent!="":
			ll_sent=len(sent.split(' '))
			if cur_len+ll_sent<=max_len:
				new_sent+=sent+"."
				cur_len+=ll_sent
			elif cur_len==0:
				new_sent+=sent+"."
				cur_len+=ll_sent
			else:
				nn=new_sent.strip()
				df.at[i, 'sentence']=nn
				break


df.to_csv(split, sep='\t')
