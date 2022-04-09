import pandas as pd
from collections import Counter
import json
from sklearn import preprocessing
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from thundersvm import SVC

#for all dataset
#df = pd.read_csv('../../Dataset/global_database_figer.csv')
df = pd.read_csv('../../Dataset/global_MFT_dataset.csv')
df = df.sample(n=1800000)

with open('../../Dataset/global_entity_name_conceptNET_embeddings.json', 'r') as f:
    data = json.load(f)

entities = list(df['entity_name'].values)
coarse_grained = list(df['global coarse grained'].values)
X = []
out_of_vocab = 0
in_vocab = 0
y = []

for entity, coarse_grain in zip(entities,coarse_grained):
    if entity not in data:
        out_of_vocab+=1
    else:
        in_vocab+=1
        X.append(data[entity])
        y.append(coarse_grain)

le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_encoded, train_size=0.80, test_size=0.20, random_state=101)
rbf = SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)

y_pred = rbf.predict(X_test)

macro = f1_score(y_test, y_pred, average='macro')
micro = f1_score(y_test, y_pred, average='micro')
weighted = f1_score(y_test, y_pred, average='weighted')

print("macro: ", macro)
print("micro: ", micro)
print("weighted: ", weighted)
