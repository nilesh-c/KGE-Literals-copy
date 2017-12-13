import pandas as pd
import pdb



literal_data = pd.read_csv('../data/fb15k-literal/entity_numerical_literal.csv',index_col=0)
literal_count = pd.read_csv('../data/fb15k-literal/frequency-count-numerical.txt', header = None, sep = '\t')

literal_count = literal_count.values
literal_map = {literal_count[idx,0]:literal_count[idx,1] for idx in range(len(literal_count))}
## Literal filtering
filtered_predicate = {predicate : count for predicate, count in literal_map.items() if (('http://rdf.freebase.com/key/' not in predicate) and count>5)}	
print('Predicate after filtering', len(filtered_predicate))
drop_columns = list(set(literal_data.columns) - set(filtered_predicate.keys())-{'Unnamed: 0'})
literal_data = literal_data.drop(drop_columns, axis=1)

literal_data.to_csv('../data/fb15k-literal/entity_filtered_literal.csv', sep=',')
df = pd.Series(filtered_predicate)
df.to_csv('../data/fb15k-literal/filtered-numerical-frequency-count.txt', sep='\t', header=None)


