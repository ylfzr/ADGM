import numpy as np
valid_idx = np.random.choice(np.arange(240000), size=40000, replace=False)
train_idx = np.arange(240000)[~valid_idx]

idx_enlarge_train = np.random.choice(np.arange(10), size=2, replace=False)
idx_tile_train = np.arange(10)[~idx_enlarge_train]

print valid_idx.shape
print train_idx.shape


print idx_enlarge_train
print idx_tile_train
print ~idx_enlarge_train