import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import spacy

import os
import unicodedata


idx2ent = np.load('data/yago3-10-literal/bin/idx2ent.npy')
ent2idx = {e: idx for idx, e in enumerate(idx2ent)}

df = pd.read_csv('data/yago3-10-literal/text_literals.txt', header=None, sep='\t')
nlp = spacy.load('en')

features = np.zeros([len(idx2ent), 384])

i = 0

for ent, txt in zip(df[0].values, df[1].values):
    vec = nlp(txt).vector
    idx = ent2idx[unicodedata.normalize('NFC', ent)]
    features[idx, :] = vec

    if i % 100 == 0:
        print(i)

    i += 1

# Save features
print('Saving text features of size {}'.format(features.shape))
np.save('data/yago3-10-literal/bin/text_literals.npy', features)
