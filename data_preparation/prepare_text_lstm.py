from text_data_utils import *
from scipy.sparse  import save_npz, csc_matrix
import numpy as np
import pickle

def triple2idx(X, idx2entity, entity_textid, empty, max_len):
    # Split Text literals
    empty_vec = np.array([empty]*max_len)
    text_s = []
    text_o = []
    for triple in X:
        if idx2entity[triple[0]] in entity_textid:
            text_s.append(entity_textid[idx2entity[triple[0]]])
        else:
            text_s.append(empty_vec)
        if idx2entity[triple[2]] in entity_textid:
            text_o.append(entity_textid[idx2entity[triple[2]]])
        else:
            text_o.append(empty_vec)

    return np.array(text_s), np.array(text_o)

# Load Text Literals
triples =[line.strip().split('\t') for line in open('data/fb15k-literal/filtered-string-literal-fb15k.txt','r')]
textliteral_id_value_map = {}
for triple in triples:
    if 'http://rdf.freebase.com/ns/common.topic.description' in triple[1]:
        entity = triple[0].replace('<http://rdf.freebase.com/ns/','')[:-1]
        if entity in textliteral_id_value_map:
            textliteral_id_value_map[entity] += triple[2]
        else:
            textliteral_id_value_map[entity] = triple[2]

text_data = list(textliteral_id_value_map.values())
vocabulary = prepare_vocab(text_data)
text_data = build_data(text_data, vocabulary, max_len=50)
entity_textid = {entity:text for entity,text in zip(textliteral_id_value_map.keys(),text_data)}

with open('data/fb15k-literal/vocabulary_text.pickle', 'wb') as f:
    pickle.dump(vocabulary, f, protocol=pickle.HIGHEST_PROTOCOL)

idx2entity = np.load('data/fb15k-literal/bin/idx2ent.npy')
# Prepare Literal-Data-Train
X_train = np.load('data/fb15k-literal/bin/train.npy')
train_text_s, train_text_o = triple2idx(X_train, idx2entity, entity_textid, empty=vocabulary['<PAD/>'], max_len=50)
train_text_s = csc_matrix(train_text_s)
train_text_o = csc_matrix(train_text_o)
save_npz('data/fb15k-literal/bin/train_text_s.npz',train_text_s)
save_npz('data/fb15k-literal/bin/train_text_o.npz',train_text_o)

# Prepare Literal-Data-Val
X_val = np.load('data/fb15k-literal/bin/val.npy')
val_text_s, val_text_o = triple2idx(X_val, idx2entity, entity_textid, empty=vocabulary['<PAD/>'], max_len=50)
val_text_s = csc_matrix(val_text_s)
val_text_o = csc_matrix(val_text_o)
save_npz('data/fb15k-literal/bin/val_text_s.npz',val_text_s)
save_npz('data/fb15k-literal/bin/val_text_o.npz',val_text_o)

# Prepare Literal-Data-Test
X_test = np.load('data/fb15k-literal/bin/test.npy')
test_text_s, test_text_o = triple2idx(X_test, idx2entity, entity_textid, empty=vocabulary['<PAD/>'], max_len=50)
test_text_s = csc_matrix(test_text_s)
test_text_o = csc_matrix(test_text_o)
save_npz('data/fb15k-literal/bin/test_text_s.npz',test_text_s)
save_npz('data/fb15k-literal/bin/test_text_o.npz',test_text_o)
