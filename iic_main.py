# khai bao bien
import tensorflow as tf  
import numpy as np 
import copy 
import time 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from data import data_load 
from graphs import IICGraph, VGG, KERNEL_INIT, BIAS_INIT 
from utils import unsupervised_labels, save_performance 
# general variables 
DPI = 600 

class ClusterIIC(object):
    def __init__(self, num_classes, learning_rate, num_repeats, save_dir = None):
        self.k_A = 5 * num_classes 
        self.num_A_sub_head = 1
        self.k_B = num_classes 
        self.num_B_sub_head = 5 
        self.num_repeats = num_repeats

        #init loss 
        self.loss_A = None 
        self.loss_B = None 
        self.losses = []

        #init output 
        self.y_hats = None 

        #init optimizer 
        # self.is_training = tf.compat.v1.placeholder(tf.bool)
        self.learning_rate = learning_rate 
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_ops = []

        #init performance dictionary
        self.perf = None 
        self.save_dir = save_dir 

        # configure performance plotting 
        self.fig_learn , self.ax_learn = plt.subplots(1,2)

    def train(self,graph, TRAIN_SET, TEST_SET, num_epochs = 10):
        1
# data 
DATA_SET = 'mnist'

DS_CONFIG = {
    'mnist':{
        'batch_size': 700,
        'num_repeats': 5,
        'mdl_input_dim': [24, 24, 1]
    }
}

TRAIN_SET, TEST_SET, SET_INFO = data_load(db_name= DATA_SET,with_info= True, **DS_CONFIG[DATA_SET])

# configure the common model elements
MDL_CONFIG = {
    # mist hyper-parameters
    'mnist': {
        'num_classes': SET_INFO.features['label'].num_classes,
        'learning_rate': 1e-4,
        'num_repeats': DS_CONFIG[DATA_SET]['num_repeats'],
        'save_dir': None},
}
# build model 
mdl = ClusterIIC(**MDL_CONFIG[DATA_SET])

# training 
mdl.train(IICGraph(config= 'B', batch_norm= True, fan_out_init= 64), TRAIN_SET, TEST_SET, num_epochs= 10)
# evaluate 
print("All done!")
