#data from https://zenodo.org/record/5514203?ref=hackernoon.com#.Y1LyQblE0UE
import pandas as pd
#
# df=pd.read_csv('train_v0.2.csv')
# df=df.rename(columns={"text": "sentence"})
# print(df)
#
# dev=df.sample(frac=0.05, axis=0)
# train=df.drop(index=dev.index)
# train=train.reset_index()
# dev=dev.reset_index()
#
#
# train.to_csv('train.tsv', sep='\t')
# dev.to_csv('dev.tsv', sep='\t')


df=pd.read_csv('test_v0.2.csv')
df=df.rename(columns={"text": "sentence"})
print(df)

df.to_csv('test.tsv', sep='\t')
