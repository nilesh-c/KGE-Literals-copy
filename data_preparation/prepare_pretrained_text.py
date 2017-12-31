import os
import sys
import time
import numpy as np
import pickle
import gensim
from data_preparation.text_data_utils import *

def load_glove(filename):
    embedding_map_pretrained = {}
    f = open('data/fb15k-literal/'+filename,'rb')
    for line in f:
        pair = line.split()
        word = pair[0]
        embedding = np.array(pair[1:], dtype='float32')
        embedding_map_pretrained[word] = embedding
    f.close()
    return embedding_map_pretrained

def pretrained_embeddings(filename='glove.6B.100d.txt', embedding_dim=100):
    # Load Text Vocabulary
    with open('data/fb15k-literal/vocabulary_text.pickle', 'rb') as f:
       vocabulary = pickle.load(f) 
    embedding_map_pretrained = load_glove(filename)
    
    embedding_weights = {}
    for word, id_ in vocabulary.items():
        if word in embedding_map_pretrained:
            embedding_weights[id_] =embedding_map_pretrained[word]
        else:
            embedding_weights[id_] = np.random.uniform(-0.25, 0.25, embedding_dim)

    with open('data/fb15k-literal/bin/pretrained-embedding.pickle', 'wb') as f:
        pickle.dump(embedding_weights, f)
