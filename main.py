import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

def loaddata():
    mnist = MNIST('/Users/cong/Downloads/MNIST') #paste the path of folder here
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)`
    y_test = np.asarray(y_test).astype(np.int32)
    X_test = np.array([[0] * 784] * 2000)
    Y_test = np.array([0] * 2000)
    X_train = np.array([[0] * 784] * 8000)
    Y_train = np.array([0] * 8000)

    for i in range(8000):
        for j in range(784):
            X_train[i,j] = x_train[i,j]
        Y_train[i] = y_train[i]
    for i in range(8000,10000):
        for j in range(784):
            X_test[i - 8000,j] = x_train[i,j]
        Y_test[i - 8000] = y_train[i]
    return (X_train,Y_train,X_test,Y_test)

def softmax_stable(Z):

    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def crossentropy_loss(Yhat, y):
    id0 = range(Yhat.shape[0])
    return -np.mean(np.log(Yhat[id0, y]))
def mlp_init(d0, d1, d2):
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros(d1)
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros(d2)
    return (W1, b1, W2, b2)
def mlp_predict(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1 # shape (N, d1)
    A1 = np.maximum(Z1, 0) # shape (N, d1)
    Z2 = A1.dot(W2) + b2 # shape (N, d2)
    return np.argmax(Z2, axis=1)
def update(W,dW,gamma,eta,v_old):
    v_new = gamma*v_old +  eta * dW
    W -= v_new
    return (W,v_new)


def mlp_fit(X, y, W1, b1, W2, b2, eta,gamma):
    loss_hist = []
    vW1= np.zeros_like(W1)
    vW2= np.zeros_like(W2)
    vb1= np.zeros_like(b1)
    vb2= np.zeros_like(b2)
    for i in range(epoch): # number of epoches
        """
        if i == 80:
            eta = 0.5
        if i == 120:
            eta = 0.2
        if i == 200:
            eta = 0.05
        """
        # feedforward
        Z1 = X.dot(W1) + b1
        A1 = softmax_stable(Z1)
        Z2 = A1.dot(W2) + b2
        Yhat = softmax_stable(Z2) # shape (N, d2)
        if i % 10 == 0: # print loss after each 10 iterations
            loss = crossentropy_loss(Yhat, y)
            print("iter %d, loss: %f" %(i, loss))
            loss_hist.append(loss)
        # back propagation
        id0 = range(Yhat.shape[0])
        Yhat[id0, y] -=1
        E2 = Yhat/N
        dW2 = np.dot(A1.T, E2)
        db2 = np.sum(E2, axis = 0)
        E1 = np.dot(E2, W2.T)
        E1[Z1 <= 0] = 0
        dW1 = np.dot(X.T, E1)
        db1 = np.sum(E1, axis = 0)
        # Gradient Descent update

        W1,tmp = update(W1,dW1,gamma,eta,vW1)
        vW1 = tmp
        b1,tmp = update(b1,db1,gamma,eta,vb1)
        vb1 = tmp
        W2,tmp = update(W2,dW2,gamma,eta,vW2)
        vW2 = tmp
        b2,tmp = update(b2,db2,gamma,eta,vb2)
        vb2 = tmp
    return (W1, b1, W2, b2, loss_hist[-1])
N = 8000
X,y,X_t,y_t = loaddata()


epoch = 600
d0 = 784 #datadimension
d1 = h = 100 #number of hidden units
d2 = C = 10 #number of classes
eta = 0.1#learning rate
gamma = 0.9
(W1, b1, W2, b2) = mlp_init(d0,d1,d2)
(W1, b1, W2, b2, loss_hist) = mlp_fit(X,y,W1,b1,W2,b2,eta,gamma)
y_pred = mlp_predict(X_t,W1,b1,W2,b2)]
acc = 100 * np.mean(y_pred == y_t)
print('training accuracy', acc,loss_hist)
