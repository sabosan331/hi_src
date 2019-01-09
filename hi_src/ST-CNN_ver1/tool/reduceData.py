#!/usr/bin/python
# coding:utf-8
# https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
import numpy as np

dataPath = '/net/xserve0/users/kojima/jikken/data/'


X1_test = np.load(dataPath + '/GS/X_test_skip.npy')
X2_test = np.load(dataPath + '/depth/X_test_depth.npy')
y_test = np.load(dataPath+'/GS/y_test_skip.npy')

X1_test = X1_test[::3]
X2_test = X2_test[::3]
y_test = y_test[::3]

np.save("X_test_GS_oneThird.npy",X1_test)
np.save("X_test_DP_oneThird.npy",X2_test)
np.save("y_test_oneThird.npy",y_test)