#####################################################################################
# Mask-based_CNN_2CL:
# Training the Mask-based CNN model using 2CL for speech enhancement,
# Given data (amplitude spectrum):
#       training_input_noisy.mat (normalized noisy speech amplitude spectrum, with zero mean and unit variance, see Fig.1)
#       validation_input_noisy.mat (normalized noisy speech amplitude spectrum, with zero mean and unit variance, see Fig.1)
#       training_pure_noise.mat
#       validation_pure_noise.mat
#       training_clean_speech.mat
#       validation_clean_speech.mat
# Output data:
#       Trained DNN model (using 2CL)
#
# Important Note: 
#
# In this script, all given data is stored in matrix with the dimensions of L*K_in (e.g. 1,000,000*132).
# K_in represents the number of input and output frequency bins and is set to 132, L 
# represents the number of frames.
# All of these matrices must be stored in version 7.3 .mat files using Matlab, by the command "save('filename.mat','variable','-v7.3')".
# Version 7.3 .mat files are used to save very large training data matrices.
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
print('  >> Loading input noisy training data... ')
mat_input = "./training_data/training_input_noisy.mat"
file_h5py_train_input = h5py.File(mat_input,'r')
train_input_noisy = file_h5py_train_input.get('training_input_noisy')
train_input_noisy = np.array(train_input_noisy)  # For converting to numpy array
train_input_noisy = np.transpose(train_input_noisy)
print('  >> Reshaping input data... ')
train_input_noisy = reshapeDataMatrix(train_input_noisy, look_backward=LOOK_BACKWARD, look_forward=LOOK_FORWARD)
print('     Input noisy training data shape: %s' % str(train_input_noisy.shape))

print('  >> Loading input noisy data for validation... ')
# Load input noisy data for validation
mat_input_vali = "./training_data/validation_input_noisy.mat"
file_h5py_vali_input = h5py.File(mat_input_vali,'r')
vali_input_noisy = file_h5py_vali_input.get('validation_input_noisy')
vali_input_noisy = np.array(vali_input_noisy)
vali_input_noisy = np.transpose(vali_input_noisy)
print('  >> Reshaping input validation data... ')
vali_input_noisy = reshapeDataMatrix(vali_input_noisy, look_backward=LOOK_BACKWARD, look_forward=LOOK_FORWARD)
print('     Input noisy validation data shape: %s' % str(vali_input_noisy.shape))


# load input data for clean noise component
print('  >> Loading clean noise training data... ')
mat_input_aux = "./training_data/training_pure_noise.mat"
file_h5py_train_input_aux = h5py.File(mat_input_aux,'r')
train_pure_noise = file_h5py_train_input_aux.get('training_pure_noise')
train_pure_noise = np.array(train_pure_noise)  # For converting to numpy array
train_pure_noise = np.transpose(train_pure_noise)
print('     Clean noise training data shape: %s' % str(train_pure_noise.shape))


# Load validation data for clean noise component
print('  >> Loading clean noise validation data... ')
mat_input_vali_aux = "./training_data/validation_pure_noise.mat"
file_h5py_vali_input_aux = h5py.File(mat_input_vali_aux,'r')
vali_pure_noise = file_h5py_vali_input_aux.get('validation_pure_noise')
vali_pure_noise = np.array(vali_pure_noise)
vali_pure_noise = np.transpose(vali_pure_noise)
print('     Clean noise validation data shape: %s' % str(vali_pure_noise.shape))


# Load clean speech training target data, which is also be used as speech component quality training input. See the first term in (5)
print('  >> Loading clean speech training data... ')
mat_target = "./training_data/training_clean_speech.mat"
training_target = h5py.File(mat_target,'r')
train_clean_speech = training_target.get('training_clean_speech')
train_clean_speech = np.array(train_clean_speech)
train_clean_speech = np.transpose(train_clean_speech)
print('     Clean speech training data shape: %s' % str(train_clean_speech.shape))

# Load clean speech validation target data, which is also be used as speech component quality validation input. See the first term in (5)
print('  >> Loading clean speech validation data... ')
mat_target_vali = "./training_data/validation_clean_speech.mat"
vali_target = h5py.File(mat_target_vali,'r')
vali_clean_speech = vali_target.get('validation_clean_speech')
vali_clean_speech = np.array(vali_clean_speech)
vali_clean_speech = np.transpose(vali_clean_speech)
print('     Clean speech validation data shape: %s' % str(vali_clean_speech.shape))

# Generate training target for residual noise power, which should be zero. See the second term in (5) 
print('  >> Loading traning target for residual noise... ')
train_residual_noise_power_target = np.zeros_like(train_pure_noise)
print('     Training residual noise target data shape: %s' % str(train_residual_noise_power_target.shape))


# Generate validation target for residual noise power, which should be zero. See the second term in (5) 
print('  >> Loading validation target for residual noise... ')
vali_residual_noise_power_target = np.zeros_like(vali_pure_noise)
print('     Validation residual noise target data shape: %s' % str(vali_residual_noise_power_target.shape))

print('> Data loading finished . Reshaping...')

###################################################
# 2.1 Data reshaping
###################################################

vali_clean_speech = np.reshape(vali_clean_speech, (vali_clean_speech.shape[0],vali_clean_speech.shape[1],1))
print('     vali_clean_speech shape: %s' % str(vali_clean_speech.shape))
train_clean_speech = np.reshape(train_clean_speech, (train_clean_speech.shape[0],train_clean_speech.shape[1],1))
print('     train_clean_speech shape: %s' % str(train_clean_speech.shape))

vali_residual_noise_power_target = np.reshape(vali_residual_noise_power_target, (vali_residual_noise_power_target.shape[0],vali_residual_noise_power_target.shape[1],1))
print('     vali_residual_noise_power_target shape: %s' % str(vali_residual_noise_power_target.shape))
train_residual_noise_power_target = np.reshape(train_residual_noise_power_target, (train_residual_noise_power_target.shape[0],train_residual_noise_power_target.shape[1],1))
print('     train_residual_noise_power_target shape: %s' % str(train_residual_noise_power_target.shape))

train_pure_noise = np.reshape(train_pure_noise, (train_pure_noise.shape[0],train_pure_noise.shape[1],1))
print('     train_pure_noise shape: %s' % str(train_pure_noise.shape))
vali_pure_noise = np.reshape(vali_pure_noise, (vali_pure_noise.shape[0],vali_pure_noise.shape[1],1))
print('     vali_pure_noise shape: %s' % str(vali_pure_noise.shape))


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
mask = Conv1D(1, N_cnn, padding='same', activation='linear')(m2)

# generate the filtered speech and noise component
n_tilde= Multiply()([mask,input_noise_component])
s_tilde= Multiply()([mask,input_speech_component])

model = Model(inputs=[input_noisy, input_noise_component, input_speech_component], outputs=[n_tilde, s_tilde])
model.summary()


#####################################################################################
# 4 Training setting
#####################################################################################

nb_epochs = 100   #100
batch_size = 128 #16
learning_rate = 5e-4 # 5e-4
adam_wn = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# the loss_weights are corresponding the weighting factors in (6), with the order [\alpha, 1-\alpha]
model.compile(optimizer=adam_wn, loss='mean_squared_error', loss_weights=[0.5, 0.5], metrics=['accuracy'])


# Stop training after 10 epoches if the vali_loss not decreasing
stop_str = cbs.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Reduce learning rate when stop improving lr = lr*factor
reduce_LR = cbs.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

best_weights = './training_results/mask_CNN_2CL' + '.h5'
best_weights = os.path.normcase(best_weights)
model_save = cbs.ModelCheckpoint(best_weights, monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=True, period=1)


#####################################################################################
# 5 Fit the model
#####################################################################################

start = time.time()

print("> Training model " + "using Batch-size: " + str(batch_size) + ", Learning_rate: " + str(learning_rate) + "...")
hist = model.fit([train_input_noisy, train_pure_noise, train_clean_speech], [train_residual_noise_power_target, train_clean_speech], epochs=nb_epochs, verbose=2, batch_size=batch_size, shuffle=True, initial_epoch=0,
                      callbacks=[reduce_LR, stop_str, model_save],
                      validation_data=[[vali_input_noisy,vali_pure_noise,vali_clean_speech], [vali_residual_noise_power_target, vali_clean_speech]]
                      )

ValiLossVec='./training_results/mask_CNN_2CL' + '_validationloss.mat'
ValiLossVec = os.path.normcase(ValiLossVec) # directory
sio.savemat(ValiLossVec, {'Vali_loss_Vec': hist.history['val_loss']})

print("> Saving Completed, Time : ", time.time() - start)
print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

