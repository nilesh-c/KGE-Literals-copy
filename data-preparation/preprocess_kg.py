import sys
sys.path.append('.')

from kga.util import *
import sys


if len(sys.argv) < 2:
    print('Please supply the directory where `train.txt`, `valid.txt`, and `test.txt` triples are in.')
    exit(1)

dataset_dir = sys.argv[1].rstrip('/')
bin_dir = '{}/bin'.format(dataset_dir)

train_path = '{}/train.txt'.format(dataset_dir)
val_path = '{}/valid.txt'.format(dataset_dir)
test_path = '{}/test.txt'.format(dataset_dir)

idx2ent, idx2rel = get_dictionary(dataset_dir)

# Save dictionaries
np.save('{}/idx2ent.npy'.format(bin_dir), idx2ent)
np.save('{}/idx2rel.npy'.format(bin_dir), idx2rel)

X_train = load_data(train_path, idx2ent, idx2rel).astype(np.int32)
X_val = load_data(val_path, idx2ent, idx2rel).astype(np.int32)
X_test = load_data(test_path, idx2ent, idx2rel).astype(np.int32)

# Save preprocessed data
np.save('{}/train.npy'.format(bin_dir), X_train)
np.save('{}/val.npy'.format(bin_dir), X_val)
np.save('{}/test.npy'.format(bin_dir), X_test)
