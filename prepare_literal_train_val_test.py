import pdb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def literal_subject_object(X, idx2literal, literal_data):
	X_literal_s = []
	X_literal_o = []
	for triple in X:
		idx_s = np.where(idx2literal==triple[0])[0]
		idx_o = np.where(idx2literal==triple[2])[0]

		X_literal_s.append(literal_data[idx_s].reshape(-1,))
		X_literal_o.append(literal_data[idx_o].reshape(-1,))
	X_literal_s = np.array(X_literal_s)
	X_literal_o = np.array(X_literal_o)

	return X_literal_s, X_literal_o
# Prepare training, validation and test file for literal dataset
# Load Literal dataset
literal_data = pd.read_csv('data/fb15k-literal/entity_literal.csv')
# For filtering Drop Columns using frequency count
idx2entity = np.load('data/fb15k-literal/bin/idx2ent.npy')
literal_data = literal_data.values
literals = literal_data[:,0]
idx2literal = np.array([np.where(entity==literals)[0] for entity in idx2entity])
idx2literal = idx2literal.reshape(-1,)
np.save('data/fb15k-literal/bin/idx2literal.npy',idx2literal) 
literal_data = literal_data[:,1:] 

# Training Set
X = np.load('data/fb15k-literal/bin/train.npy')
train_literal_s, train_literal_o = literal_subject_object(X, idx2literal, literal_data)
scaler1 = MinMaxScaler()
scaler1.fit(train_literal_s)
train_literal_s = scaler1.transform(train_literal_s)
scaler2 = MinMaxScaler()
scaler2.fit(train_literal_o)
train_literal_o = scaler2.transform(train_literal_o)

np.save('data/fb15k-literal/bin/train_literal_s.npy',train_literal_s)
np.save('data/fb15k-literal/bin/train_literal_o.npy',train_literal_o)

#Validation Set
X = np.load('data/fb15k-literal/bin/val.npy')
val_literal_s, val_literal_o = literal_subject_object(X, idx2literal, literal_data)
val_literal_s = scaler1.transform(val_literal_s)
val_literal_o = scaler2.transform(val_literal_o)

np.save('data/fb15k-literal/bin/val_literal_s.npy',val_literal_s)
np.save('data/fb15k-literal/bin/val_literal_o.npy',val_literal_o)

#Test Set
X = np.load('data/fb15k-literal/bin/test.npy')
test_literal_s, test_literal_o = literal_subject_object(X, idx2literal, literal_data)
test_literal_s = scaler1.transform(test_literal_s)
test_literal_o = scaler2.transform(test_literal_o)

np.save('data/fb15k-literal/bin/test_literal_s.npy',test_literal_s)
np.save('data/fb15k-literal/bin/test_literal_o.npy',test_literal_o)
