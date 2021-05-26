import numpy as np
import random
from tqdm import tqdm
from tqdm import trange
import cv2
import logging 

class MLP():

    def __init__(self, d0, d1, d2):
        '''
        number of the units of layers
        '''
        logging.info('Initiate model ...')
        self.W1 = 0.01 * np.random.randn(d0, d1)
        self.b1 = np.zeros(d1)
        self.W2 = 0.01 * np.random.randn(d1, d2)
        self.b2 = np.zeros(d2)

    def save_checkpoint(self, dir):

        from numpy import save
        import os

        logging.info('Saving model')
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        save(os.path.join(dir,'W1.npy'), self.W1)
        save(os.path.join(dir,'W2.npy'), self.W2)
        save(os.path.join(dir,'b1.npy'), self.b1)
        save(os.path.join(dir,'b2.npy'), self.b2)

    def load_checkpoint(self, dir):

        from numpy import load
        import os
        logging.info('Loading model from checkpoint')
        self.W1 = load(os.path.join(dir, 'W1.npy'))
        self.W2 = load(os.path.join(dir, 'W2.npy'))
        self.b1 = load(os.path.join(dir, 'b1.npy'))
        self.b2 = load(os.path.join(dir, 'b2.npy'))

    def softmax_stable(self, Z):

        e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
        A = e_Z / e_Z.sum(axis = 1, keepdims = True)
        return A

    def crossentropy_loss(self, Yhat, y):
        id0 = range(Yhat.shape[0])
        return -np.mean(np.log(Yhat[id0, y]))
    
    def test(self, dir):
        logging.info('Predicting ...')
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28 ,28), interpolation = cv2.INTER_AREA)
        cv2.imwrite("img2.png", img)
        img = np.array(img).reshape(1, 784)
        result = self.predict(img)
        return result

    def predict(self, X):
        Z1 = X.dot(self.W1) + self.b1 # shape (N, d1)
        A1 = np.maximum(Z1, 0) # shape (N, d1)
        Z2 = A1.dot(self.W2) + self.b2 # shape (N, d2)
        return np.argmax(Z2, axis=1)

    def feed_forward(self, X):

        Z1 = X.dot(self.W1) + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1.dot(self.W2) + self.b2
        Yhat = self.softmax_stable(Z2) # shape (N, d2)
        return Yhat
    def update(self, W, dW, gamma, eta, v_old):
        v_new = gamma*v_old +  eta * dW
        W -= v_new
        return (W,v_new)


    def fit(self, img, lb, epochs, eta,gamma, dur, batch, num_iters):
        N = img.shape[0]
        loss_hist = []
        vW1= np.zeros_like(self.W1)
        vW2= np.zeros_like(self.W2)
        vb1= np.zeros_like(self.b1)
        vb2= np.zeros_like(self.b2)

        acc, loss = -1, -1
        cur = 1

        logging.info('Start training with learning rate = {}:'.format(eta))
        for epoch in range(epochs): # number of epoches
            
            if (epoch / dur > cur):
                eta = eta / 1.4
                cur = i / dur + 1
                vW1, vb1, vW2, vb2 = 0, 0, 0, 0
                print("learning rate :", eta)

            pbar = tqdm(range(num_iters))
            for i in pbar:
                batch_id = random.choices(range(0, N), k = batch)
                X = np.array([img[id] for id in batch_id])
                y = np.array([lb[id] for id in batch_id])

                # feedforward
                Z1 = X.dot(self.W1) + self.b1
                A1 = np.maximum(Z1, 0)
                Z2 = A1.dot(self.W2) + self.b2
                Yhat = self.softmax_stable(Z2) # shape (N, d2)

                if i % 10 == 0: # print loss after each 10 iterations
                    loss = self.crossentropy_loss(self.feed_forward(img), lb)
                    y_pred = self.predict(img)
                    acc = 100 * np.mean(y_pred == lb)
                    loss_hist.append(loss)
                    pbar.set_postfix(loss = loss, acc = acc)
                # back propagation
                id0 = range(Yhat.shape[0])
                Yhat[id0, y] -=1
                E2 = Yhat/N
                dW2 = np.dot(A1.T, E2)
                db2 = np.sum(E2, axis = 0)
                E1 = np.dot(E2, self.W2.T)
                E1[Z1 <= 0] = 0
                dW1 = np.dot(X.T, E1)
                db1 = np.sum(E1, axis = 0)
                # Gradient Descent update

                self.W1,tmp = self.update(self.W1,dW1,gamma,eta,vW1)
                vW1 = tmp
                self.b1,tmp = self.update(self.b1,db1,gamma,eta,vb1)
                vb1 = tmp
                self.W2,tmp = self.update(self.W2,dW2,gamma,eta,vW2)
                vW2 = tmp
                self.b2,tmp = self.update(self.b2,db2,gamma,eta,vb2)
                vb2 = tmp   
        logging.info('Done training')