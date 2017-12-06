from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import os
from time import time
from sklearn.utils import shuffle as skshuffle
import pdb
from scipy.sparse import load_npz

k = 50
gamma = 1
negative_samples = 10
nepoch = 20
average_loss = True
lr = 0.01
lr_decay_every = 20
weight_decay = 1e-4
embeddings_lambda = 0
normalize_embed = False
log_interval = 9999
checkpoint_dir = 'models/'
resume = False
use_gpu = True
randseed = 9999
mbsize = 100

# Set random seed
np.random.seed(randseed)
torch.manual_seed(randseed)

if use_gpu:
    torch.cuda.manual_seed(randseed)


# Load dictionary lookups
idx2entity = np.load('data/fb15k-literal/bin/idx2ent.npy')
idx2relation = np.load('data/fb15k-literal/bin/idx2rel.npy')
n_e = len(idx2entity)
n_r = len(idx2relation)

#Load DataSet
X_train = np.load('data/fb15k-literal/bin/train.npy')
X_val = np.load('data/fb15k-literal/bin/val.npy')

# Load Literals
train_literal_s = load_npz('data/fb15k-literal/bin/train_literal_s.npz').todense().astype(np.float32)
train_literal_o = load_npz('data/fb15k-literal/bin/train_literal_o.npz').todense().astype(np.float32)
val_literal_s = load_npz('data/fb15k-literal/bin/val_literal_s.npz').todense().astype(np.float32)
val_literal_o = load_npz('data/fb15k-literal/bin/val_literal_o.npz').todense().astype(np.float32)

n_l = train_literal_s.shape[1]
M_train = X_train.shape[0]
M_val = X_val.shape[0]

k = k
lam = embeddings_lambda
C = negative_samples
# Initialize model
#model = DistMult_literal(n_e, n_r, n_l, k, lam, gpu=use_gpu)
model = RESCAL_literal(n_e, n_r, n_l, k, lam, gpu=use_gpu)
# Training params
lr = lr
wd = weight_decay
solver = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
n_epoch = nepoch
mb_size = mbsize  # 2x with negative sampling
print_every = log_interval
checkpoint_dir = '{}/fb-15k'.format(checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/rescal_rank.bin'.format(checkpoint_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Begin training
for epoch in range(n_epoch):
    print('Epoch-{}'.format(epoch+1))
    print('----------------')

    it = 0
    # Shuffle and chunk data into minibatches
    mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

    # Anneal learning rate
    lr = lr * (0.5 ** (epoch // lr_decay_every))
    for param_group in solver.param_groups:
        param_group['lr'] = lr

    for X_mb in mb_iter:
        start = time()

        # Build batch with negative sampling
        m = X_mb.shape[0]
        # C x M negative samples
        X_neg_mb = np.vstack([sample_negatives_decoupled(X_mb, n_e, n_e)
                              for _ in range(C)])
        X_train_mb = np.vstack([X_mb, X_neg_mb])

        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])
        train_literal_s_mb = train_literal_s[X_train_mb[:,0]]
        train_literal_o_mb = train_literal_o[X_train_mb[:,2]]

        # Training step
        y = model.forward(X_train_mb, train_literal_s_mb, train_literal_o_mb)

        y_pos, y_neg = y[:m], y[m:]
        loss = model.ranking_loss(
            y_pos, y_neg, margin=1, C=C, average=average_loss
        )
        loss.backward()
        solver.step()
        solver.zero_grad()
        if normalize_embed:
            model.normalize_embeddings()

        end = time()

        # Training logs
        if it % print_every == 0:
            mrr, hits10 = eval_embeddings_rel(model, X_val, n_r, k, val_literal_s, val_literal_o)
            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mrr: {:.4f}; val_hits@1: {:.4f}; time per batch: {:.2f}s'
                  .format(it, loss.data[0], mrr, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
