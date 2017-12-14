import sys
sys.path.append('.')

from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import argparse
import os
from time import time
from sklearn.utils import shuffle as skshuffle


parser = argparse.ArgumentParser(
    description='Train ERMLP with literals on MovieLens'
)

parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--mlp_h', type=int, default=100, metavar='',
                    help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--gamma', type=float, default=1, metavar='',
                    help='Ranking loss margin (default: 1)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--negative_samples', type=int, default=10, metavar='',
                    help='number of negative samples per positive sample  (default: 10)')
parser.add_argument('--nepoch', type=int, default=20, metavar='',
                    help='number of training epoch (default: 20)')
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
parser.add_argument('--use_gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--randseed', default=9999, type=int, metavar='',
                    help='resume the training from latest checkpoint (default: False')
parser.add_argument('--use_user_lit', default=False, action='store_true',
                    help='whether to use users literals (default: False)')
parser.add_argument('--use_movie_lit', default=False, action='store_true',
                    help='whether to use movies literals (default: False)')
parser.add_argument('--use_image_lit', default=False, action='store_true',
                    help='whether to use images literals (default: False)')
parser.add_argument('--use_text_lit', default=False, action='store_true',
                    help='whether to use texts literals (default: False)')

args = parser.parse_args()


# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.use_gpu:
    torch.cuda.manual_seed(args.randseed)


# Load dictionary lookups
idx2user = np.load('data/ml-100k/bin/idx2user.npy')
idx2rating = np.load('data/ml-100k/bin/idx2rating.npy')
idx2movie = np.load('data/ml-100k/bin/idx2movie.npy')

n_usr = len(idx2user)
n_rat = len(idx2rating)
n_mov = len(idx2movie)

# Load dataset
X_train = np.load('data/ml-100k/bin/rating_train.npy')
X_val = np.load('data/ml-100k/bin/rating_val.npy')

# Load literals
X_lit_usr = np.load('data/ml-100k/bin/user_literals.npy').astype(np.float32)
X_lit_mov = np.load('data/ml-100k/bin/movie_literals.npy').astype(np.float32)
X_lit_img = np.load('data/ml-100k/bin/image_literals.npy').astype(np.float32)
X_lit_txt = np.load('data/ml-100k/bin/text_literals.npy').astype(np.float32)

# Preprocess literals


def standardize(X, mean, std):
    return (X - mean) / (std + 1e-8)


mean_usr = np.mean(X_lit_usr, axis=0)
std_usr = np.std(X_lit_usr, axis=0)
X_lit_usr = standardize(X_lit_usr, mean_usr, std_usr)

mean_mov = np.mean(X_lit_mov, axis=0)
std_mov = np.std(X_lit_mov, axis=0)
X_lit_mov = standardize(X_lit_mov, mean_mov, std_mov)

# Preload literals for validation
X_lit_usr_val = X_lit_usr[X_val[:, 0]]
X_lit_mov_val = X_lit_mov[X_val[:, 2]]
X_lit_img_val = X_lit_img[X_val[:, 2]]  # Features of movies' posters
X_lit_txt_val = X_lit_txt[X_val[:, 2]]  # Features of movies' titles

M_train = X_train.shape[0]
M_val = X_val.shape[0]

n_usr_lit = X_lit_usr.shape[1]
n_mov_lit = X_lit_mov.shape[1]

k = args.k
h_dim = args.mlp_h
lam = args.embeddings_lambda
C = args.negative_samples

# Initialize model
model = ERLMLP_MovieLens(n_usr, n_mov, n_rat, n_usr_lit, n_mov_lit, k, h_dim, args.use_gpu,
                         args.use_image_lit, args.use_text_lit)

# Training params
lr = args.lr
wd = args.weight_decay

solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
n_epoch = args.nepoch
mb_size = args.mbsize  # 2x with negative sampling
print_every = args.log_interval
checkpoint_dir = '{}/ml-100k'.format(args.checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/distmult_rank.bin'.format(checkpoint_dir)

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
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_every))
    for param_group in solver.param_groups:
        param_group['lr'] = lr

    for X_mb in mb_iter:
        start = time()

        # Build batch with negative sampling
        m = X_mb.shape[0]
        # C x M negative samples
        X_neg_mb = np.vstack([sample_negatives_decoupled(X_mb, n_usr, n_mov)
                              for _ in range(C)])

        X_train_mb = np.vstack([X_mb, X_neg_mb])
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        X_lit_usr_mb = X_lit_usr[X_train_mb[:, 0]]
        X_lit_mov_mb = X_lit_mov[X_train_mb[:, 2]]
        X_lit_img_mb = X_lit_img[X_train_mb[:, 2]]
        X_lit_txt_mb = X_lit_txt[X_train_mb[:, 2]]

        # Training step
        y = model.forward(X_train_mb, X_lit_usr_mb, X_lit_mov_mb, X_lit_img_mb, X_lit_txt_mb)

        y_pos, y_neg = y[:m], y[m:]

        loss = model.ranking_loss(
            y_pos, y_neg, margin=1, C=C, average=args.average_loss
        )

        loss.backward()
        solver.step()
        solver.zero_grad()

        if args.normalize_embed:
            model.normalize_embeddings()

        end = time()

        # Training logs
        if it % print_every == 0:
            model.eval()

            hits_ks = [1, 2]

            mr, mrr, hits = eval_embeddings_rel(model, X_val, n_rat, hits_ks,
                                                X_lit_usr_val, X_lit_mov_val,
                                                X_lit_img_val, X_lit_txt_val)

            hits1, hits2 = hits

            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mr: {:.4f}; val_mrr: {:.4f}; val_hits@1: {:.4f}; val_hits@2: {:.4f} time per batch: {:.2f}s'
                  .format(it, loss.data[0], mr, mrr, hits1, hits2, end-start))

            model.train()

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
