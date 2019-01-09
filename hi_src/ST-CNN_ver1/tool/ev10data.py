#!/usr/bin/python
# coding:utf-8

import numpy as np
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import Adam
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os.path
from sklearn.metrics import confusion_matrix, classification_report

batch_size = 128
nb_classes = 10
nb_epoch = 3

img_rows = 28
img_cols = 28


def load_model():

	f_log = './log'
	f_model = '../model/'
	model_filename = '../model/save31/cnn_model.json'
	weights_filename = '../model/save31/cnn_model03-loss1.22-acc0.84-vloss1.42-vacc0.76.hdf5'

	model = model_from_json(open(f_model+model_filename).read())
	print("------------ load_weight-------------------------- ")
	model.load_weights(os.path.join(f_model,weights_filename))
	model.summary()
	model.compile(loss='categorical_crossentropy',
               optimizer= 'adam' ,
               metrics=['accuracy'])

	return model

#model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])



def input2res(dir_num,model):

	print("------------- test-loading start:" + str(dir_num) +" ----------------------")

	datapath = "/home/koji/data/"
	datapath = '/media/aoki/559a0841-86d1-4a13-bad8-8cab1f416f10/kojima/data/test10_samp2/'

	X1_test = np.load(datapath+ '/X_testGS2_id'+ str(dir_num) + '.npy')
	X2_test = np.load(datapath+'/X_testDP2_id'+ str(dir_num) + '.npy')
	y_test = np.load(datapath+'/y_test2_id'+ str(dir_num) + '.npy')

	print(X1_test.shape)
	print(X2_test.shape)
	print(y_test.shape)

	data_size_test = y_test.shape[0]

	X1_test = X1_test.reshape(data_size_test,16,90,120,1)
	X2_test = X2_test.reshape(data_size_test,16,90,120,1)
	print("------------- test-loading end ----------------------")

	X1_test = X1_test.astype('float32')
	X2_test = X2_test.astype('float32')

	X1_test /= 255.0
	X2_test /= 255.0

	nb_classes = 9 # 10→9
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	score = model.evaluate([X1_test,X2_test], Y_test, verbose=0)
	predict = model.predict_classes([X1_test,X2_test])
	#print('Test score:', score[0])
	print('Test accuracy:', score[1])
	print ( confusion_matrix(y_test, predict) )
	print ( classification_report(y_test, predict) )

	np.savez('pred_truth.npz', y_test=y_test, predict=predict)

	return y_test,predict


model = load_model()
Y_test = np.empty(0,int)
P = np.empty(0,int)

for i in range(0,10):
	
	y_test,predict = input2res(i,model)

	Y_test =  np.append(Y_test,y_test)
	P      =  np.append(P,predict)

	#score = model.evaluate([X1_test,X2_test], Y_test, verbose=0)
	#predict = model.predict_classes([X1_test,X2_test])
	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])
	print ( confusion_matrix(Y_test, P) )
	print ( classification_report(Y_test, P) )



print(Y_test.shape)
print(P.shape)

np.save('truth.npy',Y_test)
np.save('pred.npy',P)