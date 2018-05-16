
# coding: utf-8

# In[1]:


import os
import glob
import pickle
import random
import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model, Model
from keras.optimizers import SGD,RMSprop,adam

import pickle
import pandas
import numpy as np

import keras
from keras.preprocessing import image
from keras import applications
from keras.models import Sequential
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Conv2D, Conv3D,Input, ZeroPadding3D, Reshape,LSTM
from keras.layers.convolutional import Convolution2D, Convolution3D, MaxPooling2D, ZeroPadding2D,ZeroPadding3D 
from keras.layers.core import Reshape
import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.layers import Merge 
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras import regularizers
from keras import backend as K


# In[2]:


# X = np.load("train_X_normalized_lstm_regression.pkl.npy")
# Y = np.load("train_Y_normalized_lstm_regression.pkl.npy")
X = np.load("train_X_normalized_lstm_delta.pkl.npy")
Y = np.load("train_Y_normalized_lstm_delta.pkl.npy")
X_test = np.load("test_X_normalized_lstm_delta.pkl.npy")
Y_test = np.load("test_Y_normalized_lstm_delta.pkl.npy")
# In[3]:


X = X.reshape(-1, 60,5,1)

X_test = X_test.reshape(-1,60,5,1)

# In[11]:


def simple_cnn_model(X, Y):
    model = Sequential()
    model.add(Convolution2D(filters = 64, kernel_size = (30, 3), init='he_normal', padding = 'same', activation='relu', input_shape=(60, 5, 1)))
    #model.add(MaxPooling2D(pool_size = (1,2), strides = (1,2)))
    model.add(Convolution2D(filters = 128, kernel_size = (10, 2), init='he_normal', padding = 'same', activation='relu'))
    #model.add(MaxPooling2D(pool_size = (1,5), strides = (1,5)))    
    model.add(Convolution2D(filters = 256, kernel_size = (5, 2), init='he_normal', padding = 'same', activation='relu'))
    #model.add(MaxPooling2D(pool_size = (1,2), strides = (1,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation = 'softmax'))
    
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer= 'sgd', metrics=['accuracy'])
    #earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    #datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05, 
    #            shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
    #datagen.fit(x_train)
    #model.fit_generator(datagen.flow(x_train, y_label, batch_size=32),steps_per_epoch = len(x_train)/32,
    #                    epochs=10,validation_data=(x_valid, y_valid))
    
    return model
   


# In[12]:


model = simple_cnn_model(X, Y)
model.summary()


# In[ ]:

model = load_model("CNN_1_delta.h5")
for e in range(10):	
	print("Epoch ",e,"to",e+4)
	test_result = model.evaluate(X_test, Y_test)
	print(test_result)
	res = model.fit(X, Y ,batch_size=512, epochs=5,verbose=1, shuffle=True,validation_split = 0.1)
	model.save("CNN_2_delta.h5")

# In[ ]:


#model.save("CNN_1_delta.h5")

