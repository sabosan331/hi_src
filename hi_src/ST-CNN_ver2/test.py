#!/usr/bin/python
# coding:utf-8
# https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
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
import stcnnCBModel

#dataPath = '/net/xserve0/users/kojima/jikken/data/'
dataPath1 = '/media/aoki/559a0841-86d1-4a13-bad8-8cab1f416f10/kojima/data/shuffle1228/'

import keras.backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
    print("---" + str(weights))
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        #loss = K.print_tensor(loss)
        #print("-------loss--------"+ str(loss))
        return loss
    
    return loss


def generate_arrays_from_file():


    i=0
    itr=0
    ad_epoch = 0
    init_seed = 23
    while 1: 

        X1_train = np.load(dataPath1+'/GS/X_train10_id' + str(i) + '.npy')
        X2_train = np.load(dataPath1+'/depth/X_trainDP10_id' + str(i) + '.npy')
        y_train = np.load(dataPath1+'/GS/y_train10_id' + str(i) + '.npy')

        d,h,w=16,90,120
        batch_size = 10

        data_size = y_train.shape[0]
        X1_train = X1_train.reshape(data_size,d,h,w,1)
        X2_train = X2_train.reshape(data_size,d,h,w,1)

        # print("---------------------")
        # print("id" + str(i) )
        # print(X1_train.shape)
        # print(X2_train.shape)
        # print(y_train.shape)
        # print("---------------------")
        # class_weight
        # w = [0]*9
        # for lb in range(0,9):
        #     num =  len(np.where(y_train==lb)[0])
        #     if num != 0:
        #         w[lb] = float( len(y_train) / float( num ) )
        #         w[lb] = w[lb] 
        #     else:
        #         w[lb] = 0
        #     print( str(lb) + "class:" + str(num) + ": " + str(w[lb] ) )

        # prepreprocessing
        nb_classes = 9
        X1_train = X1_train.astype('float32')
        X1_train /= 255.0
        X2_train = X2_train.astype('float32')
        X2_train /= 255.0
        y_train = np_utils.to_categorical(y_train, nb_classes)
        np.random.seed(init_seed + ad_epoch )
        # data shuffle
        ad_epoch = ad_epoch + 1
        # data shuffle
        shuffle_indices = np.random.permutation(np.arange(data_size))
        X1_train = X1_train[shuffle_indices]
        X2_train = X2_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        #Y_train = np_utils.to_categorical(y_train, 9)
        #for (x,y) in zip(X_train,y_train):
        for batch_num in range(int(data_size/batch_size) ):
            # create numpy arrays of input data
            # and labels, from each line in the file

            startId = batch_num * batch_size
            endId   = min( (batch_num+1)*batch_size,data_size )

            # print(str(endId) + "," + str(data_size) )
            # if endId > data_size:
            #     print("-------------------  !!!! ---------------------------")
            #     break

            # print(startId)
            # print(endId)
            itr=itr+1
            x1 = X1_train[startId:endId]
            x2 = X2_train[startId:endId]
            y = y_train[startId:endId]            
            yield ([x1,x2], y)

        
        if i == 3:
            # print("")
            # print("---------------------")
            # print("itration:"+str(itr))
            # print("---------------------")
            # print("")
            itr=0
            i=0
        else:
            i=i+1

        del X1_train,X2_train
        del y_train


#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
#sess = tf.Session()

print("cnn_kscgr")

#from __future__ import print_function
import numpy as np
#import cv2
import os
import sys
import time

import numpy as np

#np.random.seed(1337)  # for reproducibility



dataPath = '/media/aoki/559a0841-86d1-4a13-bad8-8cab1f416f10/kojima/data/samp10_2/'
print("------------------  load test_data ----------------------------")
# X_train = np.load('data_kscgr_15.npz')['X_train']
# y_train = np.load('data_kscgr_15.npz')['y_train']
# X_test = np.load('data_kscgr_15.npz')['X_test']
# y_test = np.load('data_kscgr_15.npz')['y_test']
X1_test = np.load(dataPath + '/GS/X_train10_id4.npy')
X2_test = np.load(dataPath + '/depth/X_trainDP10_id4.npy')
y_test = np.load(dataPath+'/GS/y_train10_id4.npy')

d,h,w=16,90,120
data_size_test = y_test.shape[0]
X1_test = X1_test.reshape(data_size_test,d,h,w,1)
X2_test = X2_test.reshape(data_size_test,d,h,w,1)
# X1_test = X1_test[::2]
# X2_test = X2_test[::2]
# y_test = y_test[::2]

shuffle_id = np.random.permutation(np.arange(data_size_test))
X1_test = X1_test[shuffle_id]
X2_test = X2_test[shuffle_id]
y_test = y_test[shuffle_id]

print(X1_test.shape)
print(X2_test.shape)
print(y_test.shape)

X1_test = X1_test.astype('float32')
X1_test /= 255.0
X2_test = X2_test.astype('float32')
X2_test /= 255.0

print(X1_test.shape[0], 'test samples')

from keras.utils import np_utils
nb_classes = 9 # 10→9
Y_test = np_utils.to_categorical(y_test, nb_classes)

# ------------------------- create model -----------------------------------------------#
from keras.optimizers import SGD,Adam,Adagrad,Nadam
#optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = Adagrad(lr=0.1)
#optimizer = Adam(lr=0.01)
#opt=Adam(lr=0.001)
#opt=Adagrad(lr=0.01)
#opt = SGD(decay=1e-6, momentum=0.9, nesterov=True)
opt = Nadam()

# patience = 100
# early_stop = EarlyStopping('val_loss', patience=patience)

model = stcnnCBModel.getModel_tmp(summary=True)
#model.compile(loss='categorical_crossentropy',
#               optimizer= opt ,
#               metrics=['accuracy'])


# class_weight
# w = [0]*9
# for i in range(0,9):
#     num =  len(np.where(y_train==i)[0])
#     w[i] = len(y_train)/ float( num )
#     w[i] = w[i] 
#     print( w[i] )


#------------------------------ train and test -------------------------------------------#

# #不均衡データへの対応→class_weightをw_i = Nall / N_iでも設ける
# class_weight = {0 : 0.3,1 : 3.3, 2 : 0.9, 3 : 0.4,4 : 2.5,
#                5 : 1.0,6 : 1.4, 7 : 3.3, 8 : 1.7}
# #class_weight = {0 : 0.03,1 : 0.33, 2 : 0.09, 3 : 0.04,4 : 0.25, 5 : 0.1,6 : 0.14, 7 : 0.33, 8 : 0.17}
# # class_weight = {0:w[0],1:w[1],2:w[2],3:w[3],4:w[4],
# #                 5:w[5],6:w[6],7:w[7],8:w[8]}

# class_weight = {0 : 3,1 : 33, 2 : 9, 3 : 4,4 : 25,
#                5 : 10,6 : 14, 7 : 33, 8 : 17}

# w = 2
# class_weight = {0 : 3,1 : 33*2, 2 : 9*2, 3 : 4*2,4 : 25*2,
#                5 : 10*2,6 : 14*2/2, 7 : 33*2, 8 : 17*2*2}

# class_weight = {0 : 100./31,1 : 100./3, 2 : 100./11, 3 : 100./25,4 : 100./4,
#                5 : 100./10,6 : 100./7, 7 : 100./3, 8 : 100./6}

# class_weight = {0 : 100./31,1 : 100./3, 2 : 100./11, 3 : 100./25,4 : 100./4,
#                5 : 100./10,6 : 100./7, 7 : 100./3, 8 : 100./6}

class_weight = np.array([100./31,100./3,100./11,100./25,100./4,100./10,100./7,100./3,100./6])
class_weight = class_weight/9
#class_weight =  class_weight/np.sum(class_weight)
#class_weight = class_weight * 9

#val_class_weight = np.array([class_weight[i] for i in y_test])
#class_weight = {0 : 0.2,1 : 1, 2 : 1, 3 : 0.2,4 : 1,
#               5 : 1,6 : 1, 7 : 1, 8 : 1}
model.compile(loss=weighted_categorical_crossentropy(class_weight),
               optimizer= opt ,
               metrics=['accuracy'])

f_log = './log'
f_model = './model'

print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join(f_model,'cnn_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join(f_model,'cnn_model.yaml'), 'w').write(yaml_string)

print(class_weight)


import keras.callbacks



tb_cb = keras.callbacks.TensorBoard(log_dir=f_log, histogram_freq=1)
cp_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(f_model,'cnn_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
cbks = [cp_cb]
# history = model.fit(X_train, Y_train, batch_size=batch_size,class_weight= class_weight, epochs=epochs, callbacks = callbacks,
#           verbose=1, validation_data=(X_test, Y_test))

#train_data_size = 5717
epochs=30
steps_per_epochs=1715
steps_per_epochs=1388
#histori = model.fit_generator(generate_arrays_from_file(),class_weight= class_weight, epochs=epochs,samples_per_epoch=1000,callbacks=callbacks,validation_data=(X_test, Y_test))

hist = model.fit_generator(generate_arrays_from_file(),class_weight=class_weight, 
         epochs=epochs,steps_per_epoch=steps_per_epochs,callbacks=cbks,verbose=1,
         validation_data=([X1_test,X2_test] , Y_test))

#hist = model.fit_generator(generate_arrays_from_file(),class_weight=class_weight, 
#        epochs=epochs,steps_per_epoch=steps_per_epochs,callbacks=cbks,verbose=1,
#        validation_data=([X1_test,X2_test] , Y_test))


from sklearn.metrics import confusion_matrix, classification_report
score = model.evaluate([X1_test,X2_test], Y_test, verbose=1)
predict = model.predict_classes([X1_test,X2_test])
#print('Test score:', score[0])
print('Test accuracy:', score[1])
print ( confusion_matrix(y_test, predict) )
print ( classification_report(y_test, predict) )

acc = hist.history["acc"]
val_acc = hist.history['val_acc']

loss = hist.history["loss"]
val_loss = hist.history["val_loss"]


np.savez('res.npz', acc=acc, val_acc=val_acc, loss=loss, val_loss=val_loss )


print('save weights')
model.save_weights(os.path.join(f_model,'cnn_model_weights.hdf5'))