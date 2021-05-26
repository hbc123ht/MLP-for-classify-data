import numpy as np
import matplotlib.pyplot as plt
from data_loader import loaddata
from model import MLP
import logging
import time

logging.basicConfig(level=logging.DEBUG)

#load data
N = 10000
M = 2000
X,y,X_t,y_t = loaddata(N, M)
time.sleep(1)

#init config
d0 = 784 #datadimension
d1 = h = 1000 #number of hidden units
d2 = C = 10 #number of classes
epoch = 20 #number of epochs
eta = 0.2#learning rate
dur = 10
gamma = 0.0
batch = 100
num_iters = 101


#init model
model = MLP(d0, d1, d2)
time.sleep(1)

#training
model.fit(X, y, epoch, eta, gamma, dur, batch, num_iters)

model.save_checkpoint('model')

logging.info('Testing with testing data:')
y_pred = model.predict(X_t)

acc = 100 * np.mean(y_pred == y_t)
print('training accuracy', acc)
