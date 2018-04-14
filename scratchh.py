import cPickle as pkl
import gzip
# dataset = 'mnist_aug.pkl'
# f = open(dataset, 'rb')
# mnist_aug = pkl.load(f)
# f.close()
# x_train = mnist_aug['x_train']
# target_train = mnist_aug['target_train']
# x_test = mnist_aug['x_test']
# target_test = mnist_aug['target_test']
#
# print x_train.shape



dataset = 'data/mnist/mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pkl.load(f)
f.close()
x_train, targets_train = train_set[0], train_set[1]
x_valid, targets_valid = valid_set[0], valid_set[1]
x_test, targets_test = test_set[0], test_set[1]

print x_train.shape, targets_train.shape