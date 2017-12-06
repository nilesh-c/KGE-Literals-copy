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
    description='Train MT-KGNN on YAGO'
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
parser.add_argument('--use_user_lit', default=False, type=bool, metavar='',
                    help='whether to use users literals (default: False)')
parser.add_argument('--use_movie_lit', default=False, type=bool, metavar='',
                    help='whether to use movies literals (default: False)')

args = parser.parse_args()


# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.use_gpu:
    torch.cuda.manual_seed(args.randseed)


# Load dictionary lookups
idx2ent = np.load('data/yago3-10-literal/bin/idx2ent.npy')
idx2rel = np.load('data/yago3-10-literal/bin/idx2rel.npy')

n_ent = len(idx2ent)
n_rel = len(idx2rel)

# Load dataset
X_train = np.load('data/yago3-10-literal/bin/train.npy').astype(int)
X_val = np.load('data/yago3-10-literal/bin/val.npy').astype(int)

# Load literals
X_lit = np.load('data/yago3-10-literal/bin/numerical_literals.npy').astype(np.float32)


# Preprocess literals


def normalize(X, minn, maxx):
    return (X - minn) / (maxx - minn + 1e-8)


max_lit, min_lit = np.max(X_lit, axis=0), np.min(X_lit, axis=0)
X_lit = normalize(X_lit, max_lit, min_lit)

# Preload literals for validation
X_lit_s_val = X_lit[X_val[:, 0]]
X_lit_o_val = X_lit[X_val[:, 2]]

M_train = X_train.shape[0]
M_val = X_val.shape[0]

n_lit = X_lit.shape[1]

k = args.k
h_dim = args.mlp_h
lam = args.embeddings_lambda
C = args.negative_samples

# Initialize model
model = MTKGNN_YAGO(n_ent, n_rel, n_lit, k, h_dim, args.use_gpu)

# Training params
lr = args.lr
wd = args.weight_decay

solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
n_epoch = args.nepoch
mb_size = args.mbsize  # 2x with negative sampling
print_every = args.log_interval
checkpoint_dir = '{}/yago'.format(args.checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/mtkgnn.bin'.format(checkpoint_dir)

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
        X_neg_mb = np.vstack([sample_negatives(X_mb, n_ent)
                              for _ in range(C)])

        X_train_mb = np.vstack([X_mb, X_neg_mb])
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        m_total = X_train_mb.shape[0]

        # Random attribute to predict for users and movies in X_train_mb
        s_attr = np.random.randint(n_lit, size=m_total)
        o_attr = np.random.randint(n_lit, size=m_total)

        # Ground truth literals
        y_true_lit_s = X_lit[X_train_mb[:, 0], s_attr]
        y_true_lit_o = X_lit[X_train_mb[:, 2], o_attr]

        y_true_lit_s = Variable(torch.from_numpy(y_true_lit_s))
        y_true_lit_o = Variable(torch.from_numpy(y_true_lit_o))

        # Training step
        y_er, y_lit_s, y_lit_o = model.forward(X_train_mb, s_attr, o_attr)
        y_er_pos, y_er_neg = y_er[:m], y_er[m:]

        # Relation net update
        loss_er = model.ranking_loss(
            y_er_pos, y_er_neg, margin=1, C=C, average=args.average_loss
        )

        loss_er.backward()
        solver.step()
        solver.zero_grad()

        # Attribute nets update
        loss_lit_s = F.mse_loss(y_lit_s, y_true_lit_s)
        loss_lit_o = F.mse_loss(y_lit_o, y_true_lit_o)
        loss_lit = loss_lit_s + loss_lit_o

        loss_lit.backward()
        solver.step()
        solver.zero_grad()

        if args.normalize_embed:
            model.normalize_embeddings()

        end = time()

        # Training logs
        if it % print_every == 0:
            loss_total = loss_er + loss_lit
            mrr, hits10 = eval_embeddings(model, X_val, n_ent, 10, 100)

            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mrr: {:.4f}; val_hits@10: {:.4f}; time per batch: {:.2f}s'
                  .format(it, loss_total.data[0], mrr, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
