import pdb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Read Literal DataSet
filename1 = 'data/fb15k-literal/train_subject_numerical_triples.txt'
filename2 = 'data/fb15k-literal/train_object_numerical_triples.txt'
triples_sub = [line.strip().split('\t') for line in open(filename1,'r')]
triples_sub_col = list(zip(*triples_sub))

triples_obj = [line.strip().split('\t') for line in open(filename2,'r')]
triples_obj_col = list(zip(*triples_obj))

all_predicate = triples_sub_col[1] + triples_obj_col[1]
predicate_freq = Counter(all_predicate)
predicate_ = list(set(all_predicate))

# Literal triples for all subject and object
triples = triples_sub + triples_obj
# Frequency of relations in literal dataset
with open('data/fb15k-literal/frequency-count.txt','w') as f:
	for entity, freq in predicate_freq.items():
		f.write(entity + '\t' + str(freq) + '\n')

# Prepare literal dataset
idx2entity = np.load('data/fb15k/bin/idx2ent.npy')
idx2rel = np.load('data/fb15k/bin/idx2rel.npy')
idx2entity = np.array([idx[1:].replace('/','.') for idx in idx2entity])
np.save('data/fb15k-literal/bin/idx2ent.npy',idx2entity)
np.save('data/fb15k-literal/bin/idx2rel.npy',idx2rel)
entity_literal = {ent: np.zeros(len(predicate_)) for ent in idx2entity}
for triple in triples:
	s_ = triple[0].replace('<http://rdf.freebase.com/ns/','')[:-1]
	p_ = triple[1]
	entity_literal[s_][predicate_.index(p_)] = triple[2]

data = [entity_literal[entity] for entity in entity_literal]
df = pd.DataFrame(data = data, index = entity_literal.keys(), columns = predicate_)
df.to_csv('data/fb15k-literal/entity_literal.csv')
