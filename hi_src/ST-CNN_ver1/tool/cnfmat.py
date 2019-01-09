#!/usr/bin/env python
# -*- coding: utf-8 -*-
print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

# import some data to play with
# read input file
# y_test = np.loadtxt("truth.csv", delimiter=",",comments='#')
# y_pred = np.loadtxt("out.txt", delimiter=",",comments='#')

res1 = np.load('./pred_truth.npz')
#res2 = np.load('./res2.npz')

y_test = res1['y_test']
y_pred = res1['predict']

# y_test = np.loadtxt("truth.csv", delimiter=",",comments='#')
# y_pred = np.loadtxt("out.txt", delimiter=",",comments='#')



class_names = ["-1000","1","2","3","4","5","6","7","8"]

# detected = []
# for elems in open(args[0]):
#     elem = elems[:-1].split(' ')
#     l = int(elem[-1])
#     if l < 0:
#         l = 0
#     detected.append(int(l))

# # read ground truth
# truth = LabelList( args[1] )

# def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blacks):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     #tick_marks = np.arange(len(y_test.target_names))
#     plt.xticks(tick_marks, rotation=45)
#     ax = plt.gca()
#     ax.set_xticklabels((ax.get_xticks() +1).astype(str))
#     plt.yticks(tick_marks)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# cm = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# print('Confusion matrix, without normalization')
# print(cm)
# fig, ax = plt.subplots()
# plot_confusion_matrix(cm,classes=class_names)

# plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.gray_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
   # plt.colorbar()
    tick_marks = np.arange(len(classes))
    # print(tick_marks)
    # print("hoge")
    plt.xticks(tick_marks, classes, rotation=45,fontsize=15)
    plt.yticks(tick_marks, classes,fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Confusion Matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, round(cm[i, j],2),
        plt.text(j, i, round(cm[i, j]*100,1),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=15)

    plt.tight_layout()
    plt.ylabel('Ground Truth',fontsize=18)
    plt.xlabel('Recognition Result',fontsize=18)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)

#class_names = ["-1000","1","2","3","4","5","6","7","8"]
class_names = ["none","breaking","mixing","baking","turning","cutting","boiling","seasoning","peeling"]

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')


#round(n,2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()