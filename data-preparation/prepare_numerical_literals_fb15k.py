import pdb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Read Literal DataSet
filename = '../data/fb15k-literal/fb15k_numerical_triples.txt'
triples = [line.strip().split('\t') for line in open(filename,'r')]
all_predicate = list(zip(*triples))[1]
predicate_freq = Counter(all_predicate)

predicate_ = list(set(all_predicate))
print('Number of Unique Predicates', len(predicate_))
with open('../data/fb15k-literal/frequency-count-numerical.txt','w') as f:
	for entity, freq in predicate_freq.items():
		f.write(entity + '\t' + str(freq) + '\n')

# Prepare literal dataset
idx2entity = np.load('../data/fb15k/bin/idx2ent.npy')
idx2rel = np.load('../data/fb15k/bin/idx2rel.npy')
idx2entity = np.array([idx[1:].replace('/','.') for idx in idx2entity])
np.save('../data/fb15k-literal/bin/idx2ent.npy',idx2entity)
np.save('../data/fb15k-literal/bin/idx2rel.npy',idx2rel)
entity_literal = {ent: np.zeros(len(predicate_)) for ent in idx2entity}
for triple in triples:
	s_ = triple[0].replace('<http://rdf.freebase.com/ns/','')[:-1]
	p_ = triple[1]
	entity_literal[s_][predicate_.index(p_)] = triple[2]

data = [entity_literal[entity] for entity in entity_literal]
df = pd.DataFrame(data = data, index = entity_literal.keys(), columns = predicate_)
df.to_csv('../data/fb15k-literal/entity_numerical_literal.csv')
