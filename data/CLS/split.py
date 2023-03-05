import pandas as pd
from sklearn.model_selection import train_test_split

split='train.tsv'

df=pd.read_csv(split, sep='\t', index_col=0)
train, dev = train_test_split(df, test_size=0.1)


train.to_csv('train.tsv', sep='\t')
dev.to_csv('dev.tsv', sep='\t')