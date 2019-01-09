#!/usr/bin/python
# coding:utf-8
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D,Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras import initializers
from keras.initializers import he_normal
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.layers import Merge
from keras import regularizers


def stcnn_model():

	depth = 16
	height = 90
	width = 120
	

	encoder_1 = Sequential()

	#0st layer group
	encoder_1.add(Convolution3D(32, 3, 5, 5,\
							border_mode='same',
							subsample=(1, 1, 1), \
							input_shape=(depth, height, width,1),
							kernel_initializer = he_normal(seed =11) ,\
							bias_initializer = he_normal(seed = 32) ))
	encoder_1.add( BatchNormalization() )
	encoder_1.add( Activation('relu'))
	encoder_1.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid'))
	
	# 1st layer group
	encoder_1.add(Convolution3D(64, 3, 3, 3,\
							border_mode='same',
							subsample=(1, 1, 1),\
							kernel_initializer = he_normal(seed =11) ,\
							bias_initializer = he_normal(seed = 32) ))
	encoder_1.add( BatchNormalization() )
	encoder_1.add( Activation('relu'))
	encoder_1.add(MaxPooling3D(pool_size=(2, 2, 2), \
							strides=(2, 2, 2), \
							border_mode='valid'))
	#encoder_1.add(Dropout(0.1,seed = 233))

	# 2nd layer group
	encoder_1.add(Convolution3D(128, 3, 3, 3, \
	                          border_mode='same',\
	                          kernel_initializer = he_normal(seed = 34) ,\
	                          bias_initializer = he_normal(seed = 32)) )
	encoder_1.add( BatchNormalization() )
	encoder_1.add( Activation('relu'))
	encoder_1.add(MaxPooling3D(pool_size=(2, 2, 2), \
							strides=(2, 2, 2), 
	                        border_mode='valid'))
	encoder_1.add(Dropout(0.25,seed=756))

	# 3rd layer group
	encoder_1.add(Convolution3D(256, 3, 3, 3, 
	                          border_mode='same', 
	                          subsample=(1, 1, 1), \
	                          kernel_initializer = he_normal(seed = 322),\
	                          bias_initializer = he_normal(seed = 456)) )
	encoder_1.add( BatchNormalization() )
	encoder_1.add( Activation('relu'))
	encoder_1.add(MaxPooling3D(pool_size=(2, 2, 2), \
							   strides=(2, 2, 2), 
	                        border_mode='valid'))
	encoder_1.add(Dropout(0.25,seed=274))

	# 4th
	encoder_1.add(Convolution3D(256, 3, 3, 3, 
	                          border_mode='same',
	                          subsample=(1, 1, 1), \
	                          kernel_initializer = he_normal(seed = 543),\
	                          bias_initializer = he_normal(seed = 21)) )
	encoder_1.add( BatchNormalization() )
	encoder_1.add( Activation('relu'))
	encoder_1.add(MaxPooling3D(pool_size=(2, 2, 2), \
							strides=(1, 2, 2), 
	                        border_mode='valid'))
	encoder_1.add(Dropout(0.25,seed=746))
	encoder_1.add(Flatten())

	return encoder_1

def getModel_tmp(summary=False):
	nb_classes = 9

	
	encoder_1 = stcnn_model()
	encoder_2 = stcnn_model()

	decoder = Sequential()
	decoder.add(Merge([encoder_1, encoder_2], mode='concat'))
	decoder.add(Dense(2048, kernel_initializer = he_normal(seed = 78),bias_initializer = he_normal(seed = 45) ) )#128
	decoder.add( BatchNormalization() )
	decoder.add(Activation('relu'))
	decoder.add(Dropout(0.5,seed=74))
	decoder.add(Dense(2048, kernel_initializer = he_normal(seed = 876),bias_initializer = he_normal(seed = 54) )) #128
	decoder.add( BatchNormalization() )
	decoder.add(Activation('relu'))
	# without dropout
	decoder.add(Dense(nb_classes,kernel_initializer = he_normal(seed = 57),bias_initializer = he_normal(seed = 432),
	kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
	decoder.add(Activation('softmax'))

	if summary:
		print(encoder_1.summary())
		print(encoder_2.summary())
		print(decoder.summary())

	return decoder