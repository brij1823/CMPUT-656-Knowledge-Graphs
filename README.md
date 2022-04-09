# CMPUT-656-Knowledge-Graphs
# Two approaches for Missing Entity Types in Imbalanced Datasets:
To resolve the problem of missing entity type prediction in imbalanced datasets, we are proposing two technique. . First, develop a solution for the unbalanced dataset itself. If the distribution of types in the dataset is unbalanced, a model will likely give a biased prediction favourable to the
type that appears more frequently. Therefore, in this project, we suggest reorganising a dataset for More Frequent Types (MFT) and Less Frequent
Types (LFT) regarding each type in the dataset. We will use the distribution of types within the test dataset to find the number of types K. We will use the K as a criterion for dividing the dataset into MFT and LFT. 

Second, employ a statistical model or loss manipulation on a neural network. The out-of-vocabulary has been a common problem experienced by statis-
tical embedding methods like word2vec and GloVe. To solve the problem, we aim to use ConceptNet
Numberbatch (Speer et al., 2017), a statistical embedding that combines word2vec, GloVe, and fastText to overcome the out-of-vocabulary problem. Its methodology is to remove a letter from the end of an unknown word and determine if that letter is a prefix of known terms. If this is the case, the embeddings of those known words are averaged.

For instance, suppose we want to produce the em-
beddings of an unknown word, ”upxyz,” that is not
in the dataset. As a result, ConceptNet Numberbatch will ignore the letters at the end of the phraseand look for words with the same prefix, such as
”upstairs,” ”upcoming,” etc., and produce average embeddings based on the terms. 

We will try not to use the neural network since we cannot fully explain how it reaches a prediction. However, if the performance of the statistical models is less than what we expected, we will also consider a neural network with loss manipulation according to data
distribution.


# Dataset
We want to use the FIGER (Ling and Weld, 2012) dataset for this research. It uses a two-stage process, with the first phase detecting entity references and the second step categorizing those instances. We explored the Github repository offered in the FIGER paper for obtaining the dataset, but the data
appears to be encoded in Protobuffer, making it unusable at this time, therefore we extracted the dataset from one of the cited papers [(Abhishek
et al., 2019)](https://arxiv.org/abs/1904.13178), where the authors released the FIGER dataset in JSON format. 

# Environment

* OS: CentOS 7

* CUDA version: cuda 10

* GPU stat: GeForce RTX 2080 (X4)

* libraries
1. Keras : pip install keras
2. GloVE : pip install glove_python 
3. Thunder SVM (support vector machine with gpu): pip install thundersvm-cuda10

# How to run Codes

Please download the dataset first from here: [Dataset](https://www.example.com).
Also, if you don't have GPU envirionment, executing the model for MFT and entire dataset will take you more than 2 days. Therefore, we strongly recommend you to run the code for MFT and entire dataset on a machine has GPU envirionment.

## Word2Vec
* Location: Code/Word2Vec

* LFT: You can run the code step by step from jupyter notebook named **SVM-Word2Vec (LFT).ipynb**

* MFT & All dataset:
1) Global MFT: `python svm_word2vec.py --data ../../Dataset/global_MFT_dataset.csv --emb ../../Dataset/global_MFT.pickle --cg 'global coarse grained'`
2) Local MFT: `python svm_word2vec.py --data ../../Dataset/Local_MFT_dataset.csv --emb ../../Dataset/Local_MFT.pickle --cg coarse_grain`
3) Global entire data: `python svm_word2vec.py --data ../../Dataset/Local_database_figer.csv  --emb ../../Dataset/allDataEntity.pickle --cg coarse_grain`
4) Local entire data: `python svm_word2vec.py --data ../../Dataset/global_database_figer.csv --emb ../../Dataset/allDataEntity.pickle --cg 'global coarse grained'`

## GloVE
* Location: Code/GloVE

* Global: You can run the code step by step from jupyter notebook named **SVM(glove)_global.ipynb**

* Local: You can run the code step by step from jupyter notebook named **SVM(glove)_local.ipynb**

## ConceptNet
* Location: Code/ConceptNet

* LFT: 

* MFT & All dataset:
1) Global MFT: `python svm_conceptnet_Global.py`
2) Local MFT: `python svm_conceptnet_Local.py`
3) Global entire data: Please edit the code for entire data and run `python svm_conceptnet_Global.py`
4) Local entire data: Please edit the code for entire data and run `python svm_conceptnet_Local.py`

# Results

![](/results.png)

# Email us!

If you have any question, or if the codes doesn't work, please let us know through email: sohyun2@ualberta.ca
