import pandas as pd


name='dev.tsv'

# df=pd.read_csv(name, sep='\t', names=['label', 'sentence'], usecols=[1,2])
df=pd.read_csv(name, sep='\t', index_col=0)
print(df)
print(list(df))

df=df.replace('negative', 0)
df=df.replace('neutral', 1)
df=df.replace('positive', 2)

print(df)

df.to_csv(name, sep='\t')