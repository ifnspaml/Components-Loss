#####################################################################################
# Mask-based_CNN_2CL:
# Test the Mask-based CNN model trained by 2CL for speech enhancement,
# Given data (amplitude spectrum):
#       test_input_noisy_speech.mat (normalized noisy speech amplitude spectrum, with zero mean and unit variance using the statistics collected on the training set, see Fig.1)
#       test_pure_noise.mat
#       test_clean_speech.mat
#       test_noisy_speech_unmorm.mat (unnormalized noisy speech amplitude spectrum, used for predicting enhanced speech)
#
# Output data: (amplitude spectrum):
#       test_n_tilde_2CL.mat
#       test_s_tilde_2CL.mat
#       test_s_hat_2CL.mat
#
# Important Note: 
#
# In this script, all given data is stored in matrix with the dimensions of L*K_in (e.g. 1,000*132).
# K_in represents the number of input and output frequency bins and is set to 132, L 
# represents the number of frames.
# All of these matrices must be stored in .mat files using Matlab, by the command "save('filename.mat','variable')".
# Note that in this test script, the used .mat files are NOT stored in version 7.3 .mat format!!
#
# Technische Universität Braunschweig
# Institute for Communications Technology (IfN)
# Schleinitzstrasse 22
# 38106 Braunschweig
# Germany
# 2019 - 07 - 30
# (c) Ziyi Xu
#
# Use is permitted for any scientific purpose when citing the paper:
# Z. Xu, S. Elshamy, Z. Zhao and T. Fingscheidt, "Components Loss for Neural Networks
# in Mask-Based Speech Enhancement", arXiv preprint arXiv:
# 1908.05087.
#####################################################################################
import numpy as np
np.random.seed(1337)  # for reproducibility

from numpy import random
import os
import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import *
from keras.layers import Input, Add, Multiply, Average, Activation, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras import backend as K
import keras.optimizers as optimizers
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler
import keras.callbacks as cbs
from numpy import random
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile as swave
from sklearn import preprocessing
import math
import time
from tensorflow.python.framework import ops
from keras.backend.tensorflow_backend import set_session
import h5py
from keras.optimizers import Adam
from sklearn import preprocessing
from keras.constraints import maxnorm


###################################################
# 1 Settings
###################################################
fram_length = 132      #k_in: The number of input and output frequency bins
   
###################################################
# 1.1 CNN parameters
###################################################
n1 = 60   
n2 = 120    
n3 = 60    
N_cnn = 15 
N_cnn_last = N_cnn


LOOK_BACKWARD = 2
LOOK_FORWARD = 2
INPUT_SHAPE = (fram_length,(LOOK_BACKWARD + 1 + LOOK_FORWARD))
INPUT_SHAPE2 = (fram_length,1)


# Function used for reshape input data with context frames

def reshapeDataMatrix(data_matrix, look_backward=1, look_forward=2):
    new_dim_len = look_backward + look_forward + 1
    data_matrix_out = np.zeros((data_matrix.shape[0], data_matrix.shape[1], new_dim_len))
    for i in range(0, data_matrix.shape[0]):
        for j in range(-look_backward, look_forward + 1):
            if i < look_backward:
                idx = max(i + j, 0)
            elif i >= data_matrix.shape[0] - look_forward:
                idx = min(i + j, data_matrix.shape[0] - 1)
            else:
                idx = i + j
            data_matrix_out[i, :, j + look_backward] = data_matrix[idx, :]

    return data_matrix_out
	

###################################################
# 2 Load data
###################################################

print('> Loading data... ')
# Load input noisy data
print('> Loading test noisy input... ')
mat_input ="./test_data/test_input_noisy_speech.mat"
mat_input = os.path.normcase(mat_input)
test_noisy = sio.loadmat(mat_input)
test_noisy = test_noisy['test_input_noisy_speech']
test_noisy = np.array(test_noisy)
test_noisy = reshapeDataMatrix(test_noisy, look_backward=LOOK_BACKWARD, look_forward=LOOK_FORWARD)
print('     Input noisy data shape: %s' % str(test_noisy.shape))

# load noise component
print('> Loading test noise component... ')
mat_input_noise ="./test_data/test_pure_noise.mat"
mat_input_noise = os.path.normcase(mat_input_noise)
test_noise_component = sio.loadmat(mat_input_noise)
test_noise_component = test_noise_component['test_pure_noise']
test_noise_component = np.array(test_noise_component)


# load speech component
print('> Loading test speech component... ')
mat_input_speech ="./test_data/test_clean_speech.mat"
mat_input_speech = os.path.normcase(mat_input_speech)
test_speech_component = sio.loadmat(mat_input_speech)
test_speech_component = test_speech_component['test_clean_speech']
test_speech_component = np.array(test_speech_component)


# load unnormalized noisy speech component for estimate the enhanced speech
print('> Loading test speech component... ')
mat_input_noisy_unnorm ="./test_data/test_noisy_speech_unmorm.mat"
mat_input_noisy_unnorm = os.path.normcase(mat_input_noisy_unnorm)
test_noisy_speech = sio.loadmat(mat_input_noisy_unnorm)
test_noisy_speech = test_noisy_speech['test_noisy_speech_unmorm']
test_noisy_speech = np.array(test_noisy_speech)


print('> Data loading finished . Reshaping...')

###################################################
# 2.1 Data reshaping
###################################################

test_noise_component = np.reshape(test_noise_component, (test_noise_component.shape[0],test_noise_component.shape[1],1))
print('     test_noise_component data shape: %s' % str(test_noise_component.shape))

test_speech_component = np.reshape(test_speech_component, (test_speech_component.shape[0],test_speech_component.shape[1],1))
print('     test_speech_component data shape: %s' % str(test_speech_component.shape))

test_noisy_speech = np.reshape(test_noisy_speech, (test_noisy_speech.shape[0],test_noisy_speech.shape[1],1))
print('     test_noisy_speech data shape: %s' % str(test_noisy_speech.shape))
#####################################################################################
# 3 define model
#####################################################################################

input_noisy = Input(shape=(INPUT_SHAPE))
input_noise_component= Input(shape=(INPUT_SHAPE2))
input_speech_component = Input(shape=(INPUT_SHAPE2))

c1 = Conv1D(n1, N_cnn, padding='same')(input_noisy)
c1 = LeakyReLU(0.2)(c1)
c1 = Conv1D(n1, N_cnn, padding='same')(c1)
c1 = LeakyReLU(0.2)(c1)
x = MaxPooling1D(2)(c1)

c2 = Conv1D(n2, N_cnn, padding='same')(x)
c2 = LeakyReLU(0.2)(c2)
c2 = Conv1D(n2, N_cnn, padding='same')(c2)
c2 = LeakyReLU(0.2)(c2)
x = MaxPooling1D(2)(c2)

c3 = Conv1D(n3, N_cnn, padding='same')(x)
c3 = LeakyReLU(0.2)(c3)
x = UpSampling1D(2)(c3)

c2_2 = Conv1D(n2, N_cnn, padding='same')(x)
c2_2 = LeakyReLU(0.2)(c2_2)
c2_2 = Conv1D(n2, N_cnn, padding='same')(c2_2)
c2_2 = LeakyReLU(0.2)(c2_2)

m1 = Add()([c2, c2_2])
m1 = UpSampling1D(2)(m1)

c1_2 = Conv1D(n1, N_cnn, padding='same')(m1)
c1_2 = LeakyReLU(0.2)(c1_2)
c1_2 = Conv1D(n1, N_cnn, padding='same')(c1_2)
c1_2 = LeakyReLU(0.2)(c1_2)

m2 = Add()([c1, c1_2])

# Estimated mask from noisy input
mask = Conv1D(1, N_cnn, padding='same', activation='sigmoid')(m2)

# generate the filtered speech and noise component
n_tilde= Multiply()([mask,input_noise_component])
s_tilde= Multiply()([mask,input_speech_component])

model = Model(inputs=[input_noisy, input_noise_component, input_speech_component], outputs=[n_tilde, s_tilde])
model.summary()


#####################################################################################
# 4 Test setting
#####################################################################################

nb_epochs = 100   #100
batch_size = 128 #16
learning_rate = 5e-4 # 5e-4
adam_wn = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# the loss_weights are corresponding the weighting factors in (6), with the order [\alpha, 1-\alpha]
model.compile(optimizer=adam_wn, loss='mean_squared_error', loss_weights=[0.5, 0.5], metrics=['accuracy'])

# load the trained model
model.load_weights('./training_results/mask_CNN_2CL' + '.h5')


#####################################################################################
# 5 Prediction
# 5.1 filtered speech and noise prediction (s_tilde and n_tilde)
#####################################################################################
[n_tilde,s_tilde] = model.predict([test_noisy,test_noise_component,test_speech_component])
print('     n_tilde shape: %s' % str(n_tilde.shape))
print('     s_tilde shape: %s' % str(s_tilde.shape))
recon_n_tilde = './test_results/test_n_tilde_2CL.mat'
recon_n_tilde = os.path.normcase(recon_n_tilde)
sio.savemat(recon_n_tilde, {'test_n_tilde':n_tilde})

recon_s_tilde = './test_results/test_s_tilde_2CL.mat'
recon_s_tilde = os.path.normcase(recon_s_tilde)
sio.savemat(recon_s_tilde, {'test_s_tilde':s_tilde})
#####################################################################################
# 5.2 enhanced speech prediction (s_hat)
#####################################################################################
[s_hat2,s_hat1] = model.predict([test_noisy,test_noisy_speech,test_noisy_speech]) # the two output of the network is the same for enhanced speech prediction
print('     s_hat shape: %s' % str(s_hat2.shape))
recon_s_hat = './test_results/test_s_hat_2CL.mat'
recon_s_hat = os.path.normcase(recon_s_hat)
sio.savemat(recon_s_hat, {'test_s_hat':s_hat2})




