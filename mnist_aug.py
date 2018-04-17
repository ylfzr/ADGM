import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cPickle as pickle

def rot_img(img):
    rotated = []
    for i in range(4):
        train_img = np.rot90(img, i, (1,2))
        rotated.append(train_img)
    rotated_img = np.concatenate(rotated, axis=0)
    return rotated_img

def imresize(imgs, size=(56, 56)):
    pil_ims = []
    imgs = list(imgs)
    for img in imgs:
        pil_im = Image.fromarray((img * 255).astype(np.uint8))
        # plt.imshow(pil_im)
        # plt.show()
        pil_ims.append((np.array(pil_im.resize(size)).astype(np.float64) / 255))
    return np.array(pil_ims)

def im_tile(imgs, enlarge=2):
    imgs = list(imgs)
    images_tiled = []
    for img in imgs:
        black_block = np.zeros((56, 56))
        position = [np.random.choice([0, enlarge-1]), np.random.choice([0, enlarge-1])]
        black_block[position[0] * 28:(position[0] + 1) * 28, position[1] * 28:(position[1] + 1) * 28] = img
        images_tiled.append(black_block)
    return images_tiled



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def data_generator():
    # train_set = []
    train_num = 60000
    test_num = 10000
    train_set, train_label = mnist.train.next_batch(train_num)
    test_set, test_label = mnist.test.next_batch(test_num)

    train_img = rot_img(train_set.reshape([train_num, 28, 28]))
    test_img = rot_img(test_set.reshape([test_num, 28, 28]))
    train_label = np.tile(train_label,(4, 1))
    test_label = np.tile(test_label, (4, 1))
    # until now all good

    idx_enlarge_train = np.random.choice(np.arange(240000), size=120000, replace=False)
    # idx_tile_train = np.arange(240000)[~idx_enlarge_train]
    idx_enlarge_test = np.random.choice(np.arange(40000), size=20000, replace=False)
    # idx_tile_test = np.arange(40000)[~idx_enlarge_test]

    train_x_enlarge = imresize(train_img[idx_enlarge_train])
    train_x_tile = im_tile(np.delete(train_img, idx_enlarge_train, axis=0))

    train_x_all = np.concatenate((train_x_enlarge, train_x_tile), axis=0)
    train_y_label = np.concatenate((train_label[idx_enlarge_train], np.delete(train_label, idx_enlarge_train, axis=0)), axis=0)
    train_y_label =  np.array([np.where(r==1)[0][0] for r in list(train_y_label)])
    test_x_enlarge = imresize(test_img[idx_enlarge_test])
    test_x_tile = im_tile(np.delete(test_img, idx_enlarge_test, axis=0))
    test_x = np.concatenate((test_x_enlarge, test_x_tile), axis=0)
    test_y_label = np.concatenate((test_label[idx_enlarge_test],np.delete(test_label, idx_enlarge_test, axis=0)), axis=0)
    test_y_label =  np.array([np.where(r==1)[0][0] for r in list(test_y_label)])


    # splite validation set
    valid_idx = np.random.choice(np.arange(240000), 40000, replace=False)
    # train_idx = np.arange(240000)[~valid_idx]
    # print train_idx.shape

    train_x = np.delete(train_x_all, valid_idx, axis=0)
    train_y = np.delete(train_y_label, valid_idx, axis=0)

    valid_x = train_x_all[valid_idx]
    valid_y = train_y_label[valid_idx]

    test_y = test_y_label

    print train_x.shape, valid_x.shape, test_x.shape



    # plt.imshow(train_x[10])
    # print train_y_label[10]
    # plt.show()
    # plt.imshow(train_x[100000])
    # print train_y_label[100000]
    # plt.show()
    # plt.imshow(train_x[500])
    # print train_y_label[500]
    # plt.show()
    # plt.imshow(train_x[230000])
    # print train_y_label[230000]
    # plt.show()
    # plt.imshow(train_x[230600])
    # print train_y_label[230600]
    # plt.show()
    # plt.imshow(train_x[230080])
    # print train_y_label[230080]
    # plt.show()

    return (train_x.reshape((-1, 56 * 56)).astype(np.float32), train_y.astype(np.int64)), (valid_x.reshape((-1, 56 * 56)).astype(np.float32), valid_y.astype(np.int64)), (test_x.reshape((-1, 56 * 56)).astype(np.float32), test_y.astype(np.int64))

# data_generator()





# sio.savemat('mnist_aug.mat', {'x_train':train_x, 'target_train':train_y_label, 'x_test':test_x, 'target_test':test_y_label})
# out = open("mnist_aug.pkl", "wb")
# pickle.dump({'x_train':train_x, 'target_train':train_y_label, 'x_test':test_x, 'target_test':test_y_label}, out)
# out.close()
# train_img = map(np.reshape( , [28, 28]), list(train_set))
# rotation = lambda x: x.reshape(28, 28)











