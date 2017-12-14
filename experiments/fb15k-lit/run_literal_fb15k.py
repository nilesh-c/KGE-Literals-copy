import sys
sys.path.append('.')

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

C = 5 #negative samples
nepoch = 20
average_loss = False
lr = 0.01
lr_decay_every = 20
weight_decay = 1e-4
embeddings_lambda = 0
normalize_embed = False
print_every = 10
checkpoint_dir = 'models/'
resume = False
use_gpu = True
randseed = 9999
mbsize = 100
loss_type = 'rankloss'
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


# Initialize model
#model = DistMult_literal(n_e, n_r, n_l, k, lam, gpu=use_gpu)
embedding_size = 50
model = RESCAL_literal(n_e, n_r, n_l, embedding_size, lam=embeddings_lambda, gpu=use_gpu)
# Training params
#solver = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
n_epoch = nepoch
mb_size = mbsize  # 2x with negative sampling
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
        if loss_type == 'rankloss':
            X_neg_mb = np.vstack([sample_negatives(X_mb, n_e) for _ in range(C)])
        else:
            X_neg_mb = sample_negatives(X_mb, n_e)

        X_train_mb = np.vstack([X_mb, X_neg_mb])
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])
        train_literal_s_mb = train_literal_s[X_train_mb[:,0]]
        train_literal_o_mb = train_literal_o[X_train_mb[:,2]]
        if loss_type =='logloss':
            X_train_mb, y_true_mb, train_literal_s_mb, train_literal_o_mb = skshuffle(X_train_mb, y_true_mb, train_literal_s_mb, train_literal_o_mb)
        # Training step
        y = model.forward(X_train_mb, train_literal_s_mb, train_literal_o_mb)
        if loss_type == 'rankloss':
            y_pos, y_neg = y[:m], y[m:]
            loss = model.ranking_loss(
                y_pos, y_neg, margin=1, C=C, average=average_loss
            )
        elif loss_type =='logloss':
            loss = model.log_loss(y, y_true_mb, average=average_loss)
        loss.backward()
        solver.step()
        solver.zero_grad()
        if normalize_embed:
            model.normalize_embeddings()

        end = time()

        # Training logs
        if it % print_every == 0:
            if loss_type =='logloss':
                pred = model.predict(X_train_mb, val_literal_s, val_literal_o, sigmoid=True)
                train_acc = accuracy(pred, y_true_mb)
                # Per class training accuracy
                pos_acc = accuracy(pred[:m], y_true_mb[:m])
                neg_acc = accuracy(pred[m:], y_true_mb[m:])

                # Validation accuracy
                y_pred_val = model.forward(X_val)
                y_prob_val = F.sigmoid(y_pred_val)

                if use_gpu:
                    val_acc = accuracy(y_prob_val.cpu().data.numpy(), y_val)
                else:
                    val_acc = accuracy(y_prob_val.data.numpy(), y_val)

                # Validation loss
                val_loss = model.log_loss(y_pred_val, y_val, args.average_loss)

                print('Iter-{}; loss: {:.4f}; train_acc: {:.4f}; pos: {:.4f}; neg: {:.4f}; val_acc: {:.4f}; val_loss: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], train_acc, pos_acc, neg_acc, val_acc, val_loss.data[0], end-start))
            else:
                n_sample = 100
                k = 10
                mr, mrr, hits10 = eval_embeddings(model, X_val, n_e, k, n_sample, val_literal_s, val_literal_o)
            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mr: {:.4f}; val_mrr: {:.4f}; val_hits@{}: {:.4f}; time per batch: {:.2f}s'
                  .format(it, loss.data[0], mr, mrr, k, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
torch.save(model.state_dict(), checkpoint_path)