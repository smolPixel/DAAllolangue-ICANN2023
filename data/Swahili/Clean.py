import pandas as pd
# MPhasisFrDataset.py
split='test.tsv'

df=pd.read_csv(split, sep='\t', index_col=0)

print(len(df))
df=df.dropna()
print(len(df))
df.to_csv(split, sep='\t')