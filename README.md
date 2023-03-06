#Data Augmentation for Non-English Languages

Github accompanying the submission to ICANN 2023 "On textual data augmentation methods for Non-English languages".


#How to make it work
1. Install the requirements as noted in requirements.txt
2. Select which experiment to run from the Configs folder, or create your own Config file by copying those existing
3. Execute main.py, for example: python3 main.py --config_file Configs/CLS/100/mCBERT.yaml will run the experiments for the CLS dataset with  a starting training size of 100 and the mCBERT data augmentation algorithm. 

#Datasets
Several datasets, both English and not, are already present in the github. 