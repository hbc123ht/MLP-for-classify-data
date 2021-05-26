
from mnist import MNIST
import numpy as np
import random
import logging

def loaddata(N, M):
    logging.info('Loading data ...')
    mnist = MNIST('./MNIST') #paste the path of folder here
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)
    X_test = np.array([[0] * 784] * M)
    Y_test = np.array([0] * M)

    img_id = random.choices(range(0, x_train.shape[0]), k = N)
    X_train = np.array([x_train[id] for id in img_id])
    Y_train = np.array([y_train[id] for id in img_id])
    for i in range(M):
        for j in range(784):
            X_test[i,j] = x_train[i,j]
        Y_test[i] = y_train[i]
    logging.info('Done loading')
    return (X_train,Y_train,X_test,Y_test)