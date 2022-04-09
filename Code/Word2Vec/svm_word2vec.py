import numpy as np
import pandas as pd
import gensim
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from thundersvm import SVC
import pickle
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--emb', type=str, required=True)
parser.add_argument('--cg', type=str, required=True)
args = parser.parse_args()

model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
data = pd.read_csv(args.data)
data = data.sample(n=500000)

def preprocessing(dataset):
    
    X, y = np.asarray(dataset['sentence']), np.asarray(dataset[args.cg])
    label_map = {cat:index for index,cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    x_tokenized = [[w for w in sentence.split(" ") if w != ""] for sentence in X]
   
    return x_tokenized, y_prep

class Sequencer():
    
    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix
                ):
        
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        """
        temp_vocab = Vocab which has all the unique words
        self.vocab = Our last vocab which has only most used N words.
    
        """
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}
        """
        Now we'll create a hash map (dict) which includes words and their occurencies
        """
        for word in temp_vocab:
            # 0 does not have a meaning, you can add the word to the list
            # or something different.
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))
        
        # Now we'll sort counts and while sorting them also will sort indexes.
        # We'll use those indexes to find most used N word.
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1],counts[i] = counts[i],counts[i+1]
                    indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                else:
                    cnt += 1
        
        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])
                    
    def textToVector(self,text):
        # First we need to split the text into its tokens and learn the length
        # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
        # If it's longer than the max len we'll trim from the end.
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass
        
        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(100,))
        
        return np.asarray(vec).flatten()

x_tokenized, y_prep = preprocessing(data)
Entities = np.asarray(data['entity_name'])
e_tokenized = [[w for w in str(e).split(" ") if w != ""] for e in Entities]

with open(args.emb, 'rb') as handle:
    entity_sequencer = pickle.load(handle)

print('evecs')
e_vecs = np.asarray([entity_sequencer.textToVector(" ".join(seq)) for seq in e_tokenized])

print('pca')
pca_model = PCA(n_components=100)
pca_model.fit(e_vecs)
print("Sum of variance ratios: ",sum(pca_model.explained_variance_ratio_))

e_comps = pca_model.transform(e_vecs)

x_train,x_test,y_train,y_test = train_test_split(e_comps,y_prep,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

rbf = SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x_train, y_train)
y_pred = rbf.predict(x_test)

macro = f1_score(y_test, y_pred, average='macro')
micro = f1_score(y_test, y_pred, average='micro')
weighted = f1_score(y_test, y_pred, average='weighted')

print("macro: ", macro)
print("micro: ", micro)
print("weighted: ", weighted)
