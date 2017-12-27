"""
Filtering:
------------------------------------------
- For all val/train triples (s, p, o):
    - List all o' such that (s, p, o') is in train/val/test set
    - List all s' such that (s', p, o) is in train/val/test set

- Return 2 x (M_val or M_test) x (list of variable lengths)

During evaluation:
------------------------------------------
- Use these lists for indexing the prediction result and set them to -inf
  so that they will not ranked first.
"""
import numpy as np
import tqdm
import argparse
import os


parser = argparse.ArgumentParser(
    description='Create evaluation filters for YAGO'
)

parser.add_argument('--load', default=False, action='store_true',
                    help='load temp data or not (default: False)')
parser.add_argument('--dataset', default='yago', metavar='',
                    help='which dataset in {`yago`, `fb15k`} to be used? (default: yago)')

args = parser.parse_args()


if args.dataset == 'yago':
    dataset_dir = 'yago3-10-literal'
else:
    dataset_dir = 'fb15k-literal'

if not os.path.exists('data/{}/bin/temp/'.format(dataset_dir)):
    os.makedirs('data/{}/bin/temp/'.format(dataset_dir))


# Load dict
idx2ent = np.load('data/{}/bin/idx2ent.npy'.format(dataset_dir))
n_ent = len(idx2ent)

# Load all datasets
X_train = np.load('data/{}/bin/train.npy'.format(dataset_dir))
X_val = np.load('data/{}/bin/val.npy'.format(dataset_dir))
X_test = np.load('data/{}/bin/test.npy'.format(dataset_dir))

cnt = 1
M_val = X_val.shape[0]

if args.load:
    idx_s_prime_train = np.load('data/{}/bin/temp/idx_s_prime_train.npy'.format(dataset_dir))
    idx_o_prime_train = np.load('data/{}/bin/temp/idx_o_prime_train.npy'.format(dataset_dir))
    idx_s_prime_val = np.load('data/{}/bin/temp/idx_s_prime_val.npy'.format(dataset_dir))
    idx_o_prime_val = np.load('data/{}/bin/temp/idx_o_prime_val.npy'.format(dataset_dir))
    idx_s_prime_test = np.load('data/{}/bin/temp/idx_s_prime_test.npy'.format(dataset_dir))
    idx_o_prime_test = np.load('data/{}/bin/temp/idx_o_prime_test.npy'.format(dataset_dir))
else:
    idx_s_prime_train, idx_o_prime_train = [], []
    idx_s_prime_val, idx_o_prime_val = [], []
    idx_s_prime_test, idx_o_prime_test = [], []

    print('Gathering all entity in s and o of all datasets')
    print('-----------------------------------------------')

    for e_prime in tqdm.tqdm(range(n_ent)):
        # Train
        idx_s = set(np.where(X_train[:, 0] == e_prime)[0])
        idx_s_prime_train.append(idx_s)

        idx_o = set(np.where(X_train[:, 2] == e_prime)[0])
        idx_o_prime_train.append(idx_o)

        # Val
        idx_s = set(np.where(X_val[:, 0] == e_prime)[0])
        idx_s_prime_val.append(idx_s)

        idx_o = set(np.where(X_val[:, 2] == e_prime)[0])
        idx_o_prime_val.append(idx_o)

        # Test
        idx_s = set(np.where(X_test[:, 0] == e_prime)[0])
        idx_s_prime_test.append(idx_s)

        idx_o = set(np.where(X_test[:, 2] == e_prime)[0])
        idx_o_prime_test.append(idx_o)

    np.save('data/{}/bin/temp/idx_s_prime_train.npy'.format(dataset_dir), idx_s_prime_train)
    np.save('data/{}/bin/temp/idx_o_prime_train.npy'.format(dataset_dir), idx_o_prime_train)
    np.save('data/{}/bin/temp/idx_s_prime_val.npy'.format(dataset_dir), idx_s_prime_val)
    np.save('data/{}/bin/temp/idx_o_prime_val.npy'.format(dataset_dir), idx_o_prime_val)
    np.save('data/{}/bin/temp/idx_s_prime_test.npy'.format(dataset_dir), idx_s_prime_test)
    np.save('data/{}/bin/temp/idx_o_prime_test.npy'.format(dataset_dir), idx_o_prime_test)

    print('Done and saved!')
    print()


datasets = ['val', 'test']

for dataset in datasets:
    print('Begin filtering {} set'.format(dataset))
    print('------------------------------')

    filters_s = []
    filters_o = []

    X = X_val if dataset == 'val' else X_test

    for s, p, o in tqdm.tqdm(X):
        idx_p_train = set(np.where(X_train[:, 1] == p)[0])
        idx_p_val = set(np.where(X_val[:, 1] == p)[0])
        idx_p_test = set(np.where(X_test[:, 1] == p)[0])

        idx_sp_train = idx_s_prime_train[s] & idx_p_train
        idx_po_train = idx_p_train & idx_o_prime_train[o]
        idx_sp_val = idx_s_prime_val[s] & idx_p_val
        idx_po_val = idx_p_val & idx_o_prime_val[o]
        idx_sp_test = idx_s_prime_test[s] & idx_p_test
        idx_po_test = idx_p_test & idx_o_prime_test[o]

        """
        Step:
        -----
        Given (s, p, o')
        1. Check if it come up in X_train, X_val, and X_test
        2. If one of them are true then add o'
        Repeat for (s', p, o)
        """

        s_ents = []
        o_ents = []

        for e_prime in range(n_ent):
            # subjects
            idx_spo_train = idx_s_prime_train[e_prime] & idx_po_train
            idx_spo_val = idx_s_prime_val[e_prime] & idx_po_val
            idx_spo_test = idx_s_prime_test[e_prime] & idx_po_test

            if len(idx_spo_train | idx_spo_val | idx_spo_test):
                s_ents.append(e_prime)

            # objects
            idx_spo_train = idx_sp_train & idx_o_prime_train[e_prime]
            idx_spo_val = idx_sp_val & idx_o_prime_val[e_prime]
            idx_spo_test = idx_sp_test & idx_o_prime_test[e_prime]

            if len(idx_spo_train | idx_spo_val | idx_spo_test):
                o_ents.append(e_prime)

        # Contains subject/object entities to be ignored for this validation tripl
        filters_s.append(s_ents)
        filters_o.append(o_ents)

    # Save filters
    np.save('data/{}/bin/filter_s_{}.npy'.format(dataset_dir, dataset), filters_s)
    np.save('data/{}/bin/filter_o_{}.npy'.format(dataset_dir, dataset), filters_o)

    print('Done!')
    print()

print('All done and saved!')
