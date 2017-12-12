import sys
sys.path.append('.')

import numpy as np
import spacy

import os


X = np.load('data/ml-100k/bin/movie_literals_raw.npy')
X = X[X[:, 1] == 2]  # Only takes titles features

titles = X[:, 2]

nlp = spacy.load('en')

vecs = [nlp(title[:-7]).vector if title and title != 'unknown' else np.zeros(384)
        for title in titles]

# Save features
features = np.array(vecs)  # 1682 x 384
print('Saving text features of size {}'.format(features.shape))
np.save('data/ml-100k/bin/text_literals.npy', features)
