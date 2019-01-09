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

batch_size = 128
nb_classes = 10
nb_epoch = 3

img_rows = 28
img_cols = 28

f_log = './log'
f_model = './model/'
model_filename = 'save/1217/cnn_model_1217_1.json'
weights_filename = 'save/1217/cnn_model08-loss0.05-acc0.70-vloss2.03-vacc0.34.hdf5'

# json_string = open(os.path.join(f_model, model_filename)).read()
# model = model_from_json(json_string)
#model = model_from_json(model_filename)
print("------------- test-loading start ----------------------")

datapath = "/home/koji/data/"


X1_test = np.load(datapath+ 'GS/X_test_skip.npy')
X2_test = np.load(datapath+'depth/X_test_depth.npy')
y_test = np.load(datapath+'GS/y_test_skip.npy')
data_size_test = y_test.shape[0]

X1_test = X1_test.reshape(data_size_test,16,120,160,1)
X2_test = X2_test.reshape(data_size_test,16,120,160,1)
print("------------- test-loading end ----------------------")

# data_size_test = y_test.shape[0]
# X_test = X_test.reshape(data_size_test,16,120,160,1)
# shuffle_id = np.random.permutation(np.arange(data_size_test))
# X_test = X_test[shuffle_id]
# y_test = y_test[shuffle_id]


#print(X_train.shape)
#print(y_train.shape)
print(X1_test.shape)
print(X2_test.shape)
print(y_test.shape)



X1_test = X1_test.astype('float32')
X2_test = X2_test.astype('float32')

X1_test /= 255.0
X2_test /= 255.0
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
print(X1_test.shape[0], 'test samples')

nb_classes = 9 # 10→9

Y_test = np_utils.to_categorical(y_test, nb_classes)



model = model_from_json(open(f_model+model_filename).read())

model.summary()

model.compile(loss='categorical_crossentropy',
               optimizer= 'adam' ,
               metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])


print("------------ load_weight-------------------------- ")
model.load_weights(os.path.join(f_model,weights_filename))

#score = model.evaluate([X1_test,X2_test] , y_test, verbose=0)

from sklearn.metrics import confusion_matrix, classification_report
score = model.evaluate([X1_test,X2_test], Y_test, verbose=1)
predict = model.predict_classes([X1_test,X2_test])
#print('Test score:', score[0])
print('Test accuracy:', score[1])
print ( confusion_matrix(y_test, predict) )
print ( classification_report(y_test, predict) )


# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test  = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
# X_train = X_train.astype('float32')
# X_test  = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# old_session = KTF.get_session()

# with tf.Graph().as_default():
#     session = tf.Session('')
#     KTF.set_session(session)

#     json_string = open(os.path.join(f_model, model_filename)).read()
#     model = model_from_json(json_string)

#     model.summary()

#     model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])

#     model.load_weights(os.path.join(f_model,weights_filename))

#     cbks = []

#     history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))
#     score = model.evaluate(X_test, Y_test, verbose=0)
#     print('Test score:', score[0])
#     print('Test accuracy:', score[1])

# KTF.set_session(old_session)

