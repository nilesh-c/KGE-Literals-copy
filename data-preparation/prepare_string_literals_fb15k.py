import pdb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import spacy

# Read Literal DataSet
filename = '../data/fb15k-literal/fb15k_string_triples.txt'
triples = [line.strip().split('\t') for line in open(filename,'r')]
all_predicate = list(zip(*triples))
predicate_freq = Counter(all_predicate)
length =[len(triple) for triple in triples]
predicate_ = list(set(all_predicate))

with open('../data/fb15k-literal/frequency-count-string.txt','w') as f:
	for entity, freq in predicate_freq.items():
		f.write(entity + '\t' + str(freq) + '\n')

nlp = spacy.load('en')


# Prepare literal dataset
for triple in triples:
	if predicate_freq[triple]>5:
		s_ = triple[0].replace('<http://rdf.freebase.com/ns/','')[:-1]
		p_ = triple[1]
		entity_literal[s_][predicate_.index(p_)] = triple[2]

data = [entity_literal[entity] for entity in entity_literal]
df = pd.DataFrame(data = data, index = entity_literal.keys(), columns = predicate_)
df.to_csv('../data/fb15k-literal/entity_numerical_literal.csv')

