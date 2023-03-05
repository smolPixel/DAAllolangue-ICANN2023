import pandas as pd
from collections import Set

classes=['kimataifa', 'kitaifa', 'michezo', 'afya', 'burudani', 'uchumi']

name='train.tsv'

df=pd.read_csv(name, sep='\t', index_col=0)
print(df)
# for i, row in df.iterrows():
# 	df.at[i, 'label']=classes.index(row['label'])
	# df.at[i, 'sentence']=row['sentence'].replace('\n', ' ')
df=df.dropna()
df.to_csv(name, sep='\t')

#
# labels=list(df['label'])
# labels=set(labels)
# print(labels)
