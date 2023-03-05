import xml.etree.ElementTree as ET
import pandas as pd

df=pd.DataFrame(columns=['sentence', 'label'])

split='test'
docs=[f'books/{split}.review', f'dvd/{split}.review', f'music/{split}.review']

for dd in docs:
	mytree=ET.parse(dd)
	myroot=mytree.getroot()
	for items in myroot:
		index=len(df)
		# print(items.attrib)
		# print(items.find('text').text)
		rating=items.find('rating').text
		if rating in ["1.0", "2.0"]:
			df.at[index, 'label']=0
		elif rating in ["5.0", "4.0"]:
			df.at[index, 'label']=1
		else:
			print(rating)
		df.at[index, 'sentence']=items.find('text').text

df.to_csv(f"{split}.tsv", sep='\t')