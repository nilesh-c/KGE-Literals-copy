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
    description='Train baselines methods: RESCAL, DistMult, ER-MLP, TransE'
)

parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--negative_samples', type=int, default=10, metavar='',
                    help='number of negative samples per positive sample  (default: 10)')
parser.add_argument('--nepoch', type=int, default=5, metavar='',
                    help='number of training epoch (default: 5)')
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

n_u = len(idx2user)
n_r = len(idx2rating)
n_m = len(idx2movie)

# Load dataset
X_train = np.load('data/ml-100k/bin/train.npy')
X_val = np.load('data/ml-100k/bin/val.npy')

M_train = X_train.shape[0]
M_val = X_val.shape[0]

k = args.k
lam = args.embeddings_lambda
C = args.negative_samples

# Initialize model
model = DistMultDecoupled(n_u, n_r, n_m, k, lam, gpu=args.use_gpu)

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
        X_neg_mb = np.vstack([sample_negatives3(X_mb, n_u, n_m) for _ in range(C)])

        X_train_mb = np.vstack([X_mb, X_neg_mb])
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        # Training step
        y = model.forward(X_train_mb)
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
            mrr, hits10 = eval_embeddings_rel(model, X_val, n_r, k=1)

            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mrr: {:.4f}; val_hits@2: {:.4f}; time per batch: {:.2f}s'
                  .format(it, loss.data[0], mrr, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
