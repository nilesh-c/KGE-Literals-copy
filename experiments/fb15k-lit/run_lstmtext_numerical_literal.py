import sys
sys.path.append('.')

from kga.models.literals import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import os
from time import time
from sklearn.utils import shuffle as skshuffle
import pdb
from scipy.sparse import load_npz
import argparse
import pickle

parser = argparse.ArgumentParser(
    description='Train RESCAL for numeric and text literal-dataset'
)

parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--negative_samples', type=int, default=10, metavar='',
                    help='number of negative samples per positive sample  (default: 10)')
parser.add_argument('--nepoch', type=int, default=5, metavar='',
                    help='number of training epoch (default: 5)')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='Dimension of hidden layer in ER-MLP (default: 5)')
parser.add_argument('--p', type=float, default=0.5, metavar='',
                    help='Dropout Probability (default: 0.5)')
parser.add_argument('--average_loss', default=False, action='store_true',
                    help='whether to average or sum the loss over minibatch')
parser.add_argument('--lr', type=float, default=0.01, metavar='',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_decay_every', type=int, default=20, metavar='',
                    help='decaying learning rate every n epoch (default: 20)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='',
                    help='L2 weight decay (default: 1e-4)')
parser.add_argument('--embeddings_lambda', type=float, default=0, metavar='',
                    help='prior strength for embeddings. Constraints embeddings norms to at most one  (default: 0)')
parser.add_argument('--normalize_embed', default=False, type=bool, metavar='',
                    help='whether to normalize embeddings to unit euclidean ball (default: False)')
parser.add_argument('--log_interval', type=int, default=9999, metavar='',
                    help='interval between training status logs (default: 9999)')
parser.add_argument('--checkpoint_dir', default='models/', metavar='',
                    help='directory to save model checkpoint, saved every epoch (default: models/)')
parser.add_argument('--resume', default=False, type=bool, metavar='',
                    help='resume the training from latest checkpoint (default: False')
parser.add_argument('--use_gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--randseed', default=9999, type=int, metavar='',
                    help='resume the training from latest checkpoint (default: False')
parser.add_argument('--loss_type', default='rankloss', type=str, metavar='', 
                    help='loss function of Model, two options: rankloss and logloss')
args = parser.parse_args()

embedding_size = args.k
mb_size = args.mbsize

C = args.negative_samples #negative samples
n_epoch = args.nepoch
average_loss = args.average_loss
lr = args.lr
p = args.p
lr_decay_every = args.lr_decay_every
weight_decay = args.weight_decay
h_dim = args.h_dim
embeddings_lambda = args.embeddings_lambda
normalize_embed = args.normalize_embed
print_every = args.log_interval
checkpoint_dir = args.checkpoint_dir
resume = args.resume
use_gpu = args.use_gpu
randseed = args.randseed

loss_type = 'rankloss'
# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if use_gpu:
    torch.cuda.manual_seed(args.randseed)



# Load dictionary lookups
idx2entity = np.load('data/fb15k-literal/bin/idx2ent.npy')
idx2relation = np.load('data/fb15k-literal/bin/idx2rel.npy')
n_e = len(idx2entity)
n_r = len(idx2relation)

#Load DataSet
X_train = np.load('data/fb15k-literal/bin/train.npy')
X_val = np.load('data/fb15k-literal/bin/val.npy')

# Load Numerical Literals
train_literal_s = load_npz('data/fb15k-literal/bin/train_literal_s.npz').todense().astype(np.float32)
train_literal_o = load_npz('data/fb15k-literal/bin/train_literal_o.npz').todense().astype(np.float32)
val_literal_s = load_npz('data/fb15k-literal/bin/val_literal_s.npz').todense().astype(np.float32)
val_literal_o = load_npz('data/fb15k-literal/bin/val_literal_o.npz').todense().astype(np.float32)

# Load Text Vocabulary
with open('data/fb15k-literal/vocabulary_text.pickle', 'rb') as f:
   vocabulary = pickle.load(f)
vocab_size_text = len(vocabulary)
# Pretrained-Embeddings-for-text
with open('data/fb15k-literal/bin/pretrained-embedding.pickle', 'rb') as f:
    embedding_weights = pickle.load(f)
# Load Text Literals
train_text_s = load_npz('data/fb15k-literal/bin/train_text_s.npz').toarray()
train_text_o = load_npz('data/fb15k-literal/bin/train_text_o.npz').toarray()

val_text_s = load_npz('data/fb15k-literal/bin/val_text_s.npz').toarray()
val_text_o = load_npz('data/fb15k-literal/bin/val_text_o.npz').toarray()

embedding_weights = np.array(list(embedding_weights.values()))
dim_text = embedding_weights.shape[1]
n_numeric = train_literal_s.shape[1]
M_train = X_train.shape[0]
M_val = X_val.shape[0]

# Initialize model
#model = DistMult_literal(n_e, n_r, n_l, k, lam, gpu=use_gpu)
#model = RESCAL_literal(n_e, n_r,embedding_size , embeddings_lambda, n_l, n_text, gpu=use_gpu)
batch_size = mb_size + C*mb_size
text_length = 50
model = ERMLP_literal2(n_e, n_r, embedding_size, h_dim, p, embeddings_lambda, n_numeric, vocab_size_text, dim_text, embedding_weights, batch_size, text_length, numeric = True, text=True, gpu=True)

# Training params
#solver = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
checkpoint_dir = '{}/fb-15k'.format(checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/lstm_text_ermlp_rank.bin'.format(checkpoint_dir)

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

        train_text_s_mb = train_text_s[X_train_mb[:,0]]
        train_text_o_mb = train_text_o[X_train_mb[:,2]]
        if loss_type =='logloss':
            X_train_mb, y_true_mb, train_literal_s_mb, train_literal_o_mb, train_text_s_mb, train_text_o_mb = skshuffle(X_train_mb, y_true_mb, train_literal_s_mb, train_literal_o_mb, train_text_s_mb, train_text_o_mb)       
        # Training step
        y = model.forward(X_train_mb, train_literal_s_mb, train_literal_o_mb, train_text_s_mb, train_text_o_mb)

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
                #pred = model.predict(X_train_mb, val_literal_s, val_literal_o, val_string_s, val_string_o, sigmoid=True)
                pred = model.predict(X_train_mb, val_literal_s, val_literal_o, val_text_s, val_text_o)
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
                mr, mrr, hits10 = eval_embeddings(model, X_val, n_e, k, n_sample, val_literal_s, val_literal_o, val_text_s, val_text_o)
            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mrr: {:.4f}; val_hits@10: {:.4f}; time per batch: {:.2f}s'
                  .format(it, loss.data[0], mrr, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
torch.save(model.state_dict(), checkpoint_path)