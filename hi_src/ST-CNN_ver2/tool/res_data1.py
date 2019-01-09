#!/usr/bin/python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
# import sys 
# new_path = "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages" # or wherever your pip install drops things
# sys.path.append(new_path)
#mport pandas
#import pandas as pd

def plot_history(hist):
    # 精度の履歴をプロット
    plt.plot(hist['acc'],"o-",label="accuracy")
    plt.plot(hist['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

res1 = np.load('./res.npz')
#res2 = np.load('./res2.npz')

acc1 = res1['acc']
val_acc1 = res1['val_acc']
loss1 = res1['loss']
val_loss1 = res1['val_loss']

# acc2 = res2['acc']
# val_acc2 = res2['val_acc']
# loss2 = res2['loss']
# val_loss2 = res2['val_loss']


# acc1 = np.append(acc1,acc2)
# val_acc1 = np.append(val_acc1,val_acc2)
# loss1 = np.append(loss1,loss2)
# val_loss1 = np.append(val_loss1,val_loss2)
plt.rcParams["font.size"] = 20

plt.plot(acc1,"o-",label="accuracy")
plt.plot(val_acc1,"o-",label="val_acc")
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc="lower right")
plt.show()

# 損失の履歴をプロット
plt.plot(loss1,"o-",label="loss",)
plt.plot(val_loss1,"o-",label="val_loss")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()

#print(res['acc'])
#res = Series(res)

#plot_history(res)
