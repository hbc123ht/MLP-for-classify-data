import numpy as np
import time
import matplotlib.pyplot as plt
from data_loader import loaddata
from model import MLP
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

#init config 
d0 = 784 #datadimension
d1 = h = 1000 #number of hidden units
d2 = C = 10 #number of classes

#init model
model = MLP(d0, d1, d2)
time.sleep(1)

#load model
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--check_point', default = 'model')
args = parser.parse_args()

model.load_checkpoint(args.check_point)

#load img
result = model.test(args.input_dir)
logging.info('Result is {}'.format(result))
