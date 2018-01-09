import sys
sys.path.append('.')

import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm


idx2ent = np.load('data/fb15k-literal/bin/idx2ent.npy')
n_ent = len(idx2ent)

X_train = np.load('data/fb15k-literal/bin/train.npy')
X_val = np.load('data/fb15k-literal/bin/val.npy')
X_test = np.load('data/fb15k-literal/bin/test.npy')

X = np.vstack([X_train, X_val, X_test])

lit_s_train = load_npz('data/fb15k-literal/bin/train_text_s.npz').todense()
lit_s_val = load_npz('data/fb15k-literal/bin/val_text_s.npz').todense()
lit_s_test = load_npz('data/fb15k-literal/bin/test_text_s.npz').todense()

lit_s = np.vstack([lit_s_train, lit_s_val, lit_s_test])

lit_o_train = load_npz('data/fb15k-literal/bin/train_text_o.npz').todense()
lit_o_val = load_npz('data/fb15k-literal/bin/val_text_o.npz').todense()
lit_o_test = load_npz('data/fb15k-literal/bin/test_text_o.npz').todense()

lit_o = np.vstack([lit_o_train, lit_o_val, lit_o_test])

res = np.zeros([n_ent, lit_s.shape[1]])

for x, l_s, l_o in tqdm(zip(X, lit_s, lit_o)):
    s, o = x[0], x[2]
    res[s, :] = l_s
    res[o, :] = l_o

np.save('data/fb15k-literal/bin/text_literals.npy', res)
