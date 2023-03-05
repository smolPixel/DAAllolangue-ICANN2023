import pandas as pd
from sklearn.model_selection import train_test_split

split='test.tsv'

df=pd.read_csv(split, sep='\t', usecols=[0,3])

print(set(list(df['hate'])))
df=df.replace('hate', 0)
df=df.replace('none', 1)
df=df.replace('offensive', 2)
df=df.rename(columns={'comments':'sentence', 'hate':'label'})

print(len(df))

if split=="train.tsv":
	train, dev = train_test_split(df, test_size=0.1)
	print(len(train))
	print(len(dev))
	train.to_csv('train.tsv', sep='\t')
	dev.to_csv('dev.tsv', sep='\t')
else:
	df.to_csv('test.tsv', sep='\t')
print(df)