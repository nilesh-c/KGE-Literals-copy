import pdb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import spacy
import re

# Read Literal DataSet
filename = '../data/fb15k-literal/fb15k_string_triples.txt'
triples = [line.strip().split('\t') for line in open(filename,'r')]
triples_transpose = list(zip(*triples))
entities = list(set(triples_transpose[0]))
all_predicate = triples_transpose[1]
predicate_freq = Counter(all_predicate)
length =[len(triple) for triple in triples]
predicate_ = list(set(all_predicate))

with open('../data/fb15k-literal/frequency-count-string.txt','w') as f:
	for pred, freq in predicate_freq.items():
		f.write(pred + '\t' + str(freq) + '\n')

nlp = spacy.load('en')

# Prepare literal dataset
text_literal_reprsn = {}
filtered_triples = []			
for triple in triples:
	if 'http://rdf.freebase.com/ns/common.topic.description' in triple[1] or predicate_freq[triple[1]]>5:
		s_ = triple[0].replace('<http://rdf.freebase.com/ns/','')[:-1]
		p_ = triple[1]
		literal = triple[2]
		literal = ' '.join(w for w in re.split(r"\W", literal) if w)
		if literal != 'unknown':
			literal_reprsn = nlp(literal).vector
		else:
			literal_reprsn = np.zeros(384)
		if s_ in text_literal_reprsn:
			text_literal_reprsn[s_].append(literal_reprsn)
		else:
			text_literal_reprsn[s_] = [literal_reprsn]
		filtered_triples.append(triple)

reprsn_text = []
entity_ = []
for entity,reprsn in text_literal_reprsn.items():
	reprsn_text.append(np.average(np.array(reprsn), axis=0))
	entity_.append(entity)
reprsn_text = np.array(reprsn_text)
np.save('../data/fb15k-literal/entity2stringliteral.npy', np.array(entity_))
np.save('../data/fb15k-literal/entity_string_literal_reprsn.npy', np.array(reprsn_text))	

with open('../data/fb15k-literal/filtered-string-literal-fb15k.txt') as f:
	for triple in filtered_triples:
		f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')		



