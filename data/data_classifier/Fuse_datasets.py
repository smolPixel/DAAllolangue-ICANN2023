"""Fuse all datasets compiled beforehand"""
import pandas as pd

df0=pd.read_csv('dataset_classifier_SST-2.tsv', index_col=0, sep='\t')
df1=pd.read_csv('dataset_classifier_FakeNews.tsv', index_col=0, sep='\t')
df2=pd.read_csv('dataset_classifier_Irony.tsv', index_col=0, sep='\t')

dfFinal=df0.append(df1)
dfFinal=dfFinal.append(df2)
dfFinal.to_csv('dataset.tsv', sep='\t')