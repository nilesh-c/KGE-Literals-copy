import pdb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import spacy
import re

# Read Literal DataSet
filename = 'data/fb15k-literal/fb15k_string_triples.txt'
triples = [line.strip().split('\t') for line in open(filename,'r')]
triples_transpose = list(zip(*triples))
entities = list(set(triples_transpose[0]))
all_predicate = triples_transpose[1]
predicate_freq = Counter(all_predicate)
length =[len(triple) for triple in triples]
predicate_ = list(set(all_predicate))

with open('data/fb15k-literal/frequency-count-string.txt','w') as f:
	for pred, freq in predicate_freq.items():
		f.write(pred + '\t' + str(freq) + '\n')

nlp = spacy.load('en')
# Filter literal Dataset
filtered_triples = []			
for triple in triples:
	if (('http://rdf.freebase.com/ns/common.topic.description' in triple[1] or predicate_freq[triple[1]]>5) and 'http://rdf.freebase.com/ns/common.topic.alias' not in triple[1]):
		s_ = triple[0].replace('<http://rdf.freebase.com/ns/','')[:-1]
		p_ = triple[1]
		literal = triple[2]
		literal = ' '.join(w for w in re.split(r"\W", literal) if w)
		filtered_triples.append(triple)

# Prepare literal dataset
filtered_triples_transpose = list(zip(*filtered_triples))
unique_entities = list(set(filtered_triples_transpose[0]))
unique_literal_relations = list(set(filtered_triples_transpose[1]))
text_literal_reprsn = np.zeros((len(unique_entities),len(unique_literal_relations),384), dtype='float32')
for triple in filtered_triples:
		entity = triple[0]
		literal = triple[2]
		literal_relation = triple[1]
		if literal == 'unknown':
			continue
		literal_reprsn = nlp(literal).vector
		idx_lit = unique_literal_relations.index(literal_relation)
		idx_ent = unique_entities.index(entity)
		text_literal_reprsn[idx_ent,idx_lit,:] = literal_reprsn
unique_entities = [entity.replace('<http://rdf.freebase.com/ns/','')[:-1] for entity in unique_entities]
np.save('data/fb15k-literal/entity2stringliteral.npy', np.array(unique_entities))
np.save('data/fb15k-literal/entity_string_literal_reprsn.npy', text_literal_reprsn)	

with open('sdata/fb15k-literal/filtered-string-literal-fb15k.txt','w') as f:
	for triple in filtered_triples:
		f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')		



