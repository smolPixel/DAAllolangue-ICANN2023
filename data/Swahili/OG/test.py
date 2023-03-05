import pandas as pd

df=pd.read_csv('train.tsv', sep='\t')
print(df)
print(len(df))
df=df.dropna()
print(len(df))

print(list(df['sentence']))