import numpy as np
import pickle as pkl
import cPickle as cPkl
import gzip, zipfile, tarfile
import os
import scipy.io as sio

# dataset = 'data/mnist/mnist.pkl.gz'
# print os.getcwd()
# f = gzip.open(dataset, 'rb')
# train_set, valid_set, test_set = pkl.load(f)
# f.close()
#
# x_train, targets_train = train_set[0], train_set[1]
# x_valid, targets_valid = valid_set[0], valid_set[1]
# x_test, targets_test = test_set[0], test_set[1]
#
# print x_train.shape
# print targets_train.shape


# dataset = 'mnist_aug.pkl'
# mnist_aug = pkl.load(dataset)
# x_train = mnist_aug['x_train']
# target_train = mnist_aug['target_train']
# x_test = mnist_aug['x_test']
# target_test = mnist_aug['target_test']
#
# print x_train.shape


# data1 = {'a': [1, 2.0, 3, 4+6j],
#          'b': ('string', u'Unicode string'),
#          'c': None}
#
# selfref_list = [1, 2, 3]
# selfref_list.append(selfref_list)
#
# output = open('data.pkl', 'wb')
#
# # Pickle dictionary using protocol 0.
# pkl.dump(data1, output)
#
# # Pickle the list using the highest protocol available.
# pkl.dump(selfref_list, output, -1)
#
# output.close()


a = pkl.load('data.pkl')
print a 
