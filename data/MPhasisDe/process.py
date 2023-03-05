import pandas as pd

split='dev'

df= pd.read_csv(f'de.{split}.csv', sep='\t')
df=df[['content', 'e1']]
print(len(df))
df=df[df.e1!=1]
df=df.replace(2, 1)
df=df.rename(columns={"content": "sentence", "e1": "label"})
print(df)
df.to_csv(f"{split}.tsv", sep='\t')