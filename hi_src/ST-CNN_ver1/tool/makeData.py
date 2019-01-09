#!/usr/bin/python
# coding:utf-8
import numpy as np
import cv2
import os
import sys
import time
#%matplotlib inline

# I dont know why #labels < #images

# 幹パス
# path = "/home/seiji/Data/KSCGR/" # local 
path = "/net/xserve0/users/kojima/Data/KSCGR/" # server
# 学習データディレクトリ
dirs_train = ["boild-egg-1","boild-egg-2","boild-egg-3","boild-egg-4","boild-egg-5",
            "ham-egg-1","ham-egg-2","ham-egg-3","ham-egg-4","ham-egg-5",
            "kinshi-egg-1","kinshi-egg-2","kinshi-egg-3","kinshi-egg-4","kinshi-egg-5",
            "omelette-1","omelette-2","omelette-3","omelette-4","omelette-5",
            "scramble-egg-1","scramble-egg-2","scramble-egg-3","scramble-egg-4","scramble-egg-5"]
# 評価データディレクトリ
dirs_test = ["test_data_10_01","test_data_10_02","test_data_10_03","test_data_10_04","test_data_10_05",
            "test_data_11_01","test_data_11_02","test_data_11_03","test_data_11_04","test_data_11_05"]

skip_train=10
skip_test=30
depth = 16
space = 1
b_f = depth*space/2
# s_seq = 16
# e_seq = 25
height = 480
width = 640
scale = 4
sc_w = 120
sc_h=90
ch = 1


###################### train img loading 
for idx in range(0,5): # divided into 5 data
	X_train = np.empty( (0,depth,sc_h,sc_w),float)
	y_train = np.empty(0,int)
	s_seq = idx*5 
	e_seq = s_seq+5
	for dir_num in range(s_seq,e_seq):
	    list_imgs = os.listdir(path+dirs_train[dir_num]+"/image_jpg/")
	    print(len(list_imgs))
	    x_train = np.empty( (0,depth,sc_h,sc_w),float)
	    for f_num in range(depth*space/2,len(list_imgs)-(depth*space/2)+1):
	        if f_num%skip_train == 0:
	            Img3D = np.empty( (0,sc_h,sc_w) ,float)
	            for _plus in range(0,depth):
	                img = cv2.imread(path+dirs_train[dir_num]+"/image_jpg/" + str(f_num -depth*space/2 +_plus*space) + '.jpg', 0)
	                #height,width = img.shape[0],img.shape[1]
	                #print(height,width)
	                img = cv2.resize(img,(sc_w,sc_h))
	                #img1 = cv2.resize(img,(120,90))
	                #img1 = cv2.resize(img,(112,112))
	                #cv2.imshow('wind1',img)
	            	#cv2.imshow('window', img)
	            	#cv2.waitKey(2)
	            	img = img.reshape(ch,sc_h,sc_w)
	                Img3D = np.append(Img3D,img,axis=0)
	            Img3D = Img3D.reshape(ch,depth,sc_h,sc_w)
	            x_train = np.append(x_train,Img3D,axis=0)
	            sys.stdout.write("\r%d %% (%s/%s) " % ( (100*f_num/(len(list_imgs)-1)), str(dir_num) ,25 ) )
	            sys.stdout.flush()
	            time.sleep(0.00001)
	    X_train = np.append(X_train,x_train,axis=0)
	    print(x_train.shape)
	    del x_train

	    labelfile = np.loadtxt( path+dirs_train[dir_num]+"/labels.txt", delimiter="\t",comments='#') # ラベル読み込み 
	    gomi,label = np.hsplit(labelfile,[-1]) # ラベルファイルだけ取り出し
  		 # print(len(label) )
	    label = label[b_f:-b_f:skip_train]
	    # if ( not( and dir_num!=8 and dir_num!=9 and dir_num!=11 and dir_num!=13 and dir_num!=14
	    #     and dir_num != 22 and dir_num != 23) ):
	    #     label = label[0:-1:1]
	    print(str(dir_num) + "," + str (len(label)) )
	    y_train =  np.append(y_train,label)

	print("----------------------" + str(idx) + "------------------------------")
	y_train = y_train.flatten() # model入力用に変換
	y_train[y_train==-1000]=0
	print(X_train.shape)
	print(y_train.shape)
	np.save('X_train10_id' + str(idx) + '.npy', X_train)
	np.save('y_train10_id' + str(idx) + '.npy', y_train)
	print("--------------------------------------------------------")
	del X_train,y_train


print("----------------------- start to make test datas ---------------------------------")
####################### test loading
X_test = np.empty( (0,depth,sc_h,sc_w),float)
y_test = np.empty(0,int)
for dir_num in range(0,10):
    list_imgs = os.listdir(path+dirs_test[dir_num]+"/image_jpg/")
    #print(len(list_imgs))
    x_test = np.empty( (0,depth,sc_h,sc_w),float)
    for f_num in range(depth*space/2,len(list_imgs)-(depth*space/2)+1):
        if f_num%skip_test == 0:
            Img3D = np.empty( (0,sc_h,sc_w) ,float)
            for _plus in range(0,depth):
                img = cv2.imread(path+dirs_test[dir_num]+"/image_jpg/" + str(f_num -depth*space/2 +_plus*space) + '.jpg', 0)
                #height,width = img.shape[0],img.shape[1]
                #print(height,width)
                img = cv2.resize(img,(sc_w,sc_h))
                #img1 = cv2.resize(img,(120,90))
                #img1 = cv2.resize(img,(112,112))
                #cv2.imshow('wind1',img)
            	cv2.imshow('window', img)
            	cv2.waitKey(1)
            	img = img.reshape(ch,sc_h,sc_w)
                Img3D = np.append(Img3D,img,axis=0)
            Img3D = Img3D.reshape(ch,depth,sc_h,sc_w)
            x_test = np.append(x_test,Img3D,axis=0)
            sys.stdout.write("\r%d %% (%s/%s) " % ( (100*f_num/(len(list_imgs)-1)), str(dir_num) ,25 ) )
            sys.stdout.flush()
            time.sleep(0.00001)
    X_test = np.append(X_test,x_test,axis=0)
    print(x_test.shape)
    del x_test
    labelfile = np.loadtxt( path+dirs_test[dir_num]+"/labels.txt", delimiter="\t",comments='#') # ラベル読み込み 
    gomi,label = np.hsplit(labelfile,[-1]) # ラベルファイルだけ取り出し
    label = label[b_f:-b_f:skip_test]

    if (dir_num != 3 and dir_num != 5 and dir_num != 8):
    	label = label[0:-1:1]


    print(label.shape)
    print("---- ---- ---- append ---- ---- ----")
    y_test =  np.append(y_test,label)


print(X_test.shape)
print(y_test.shape)
#print("サンプリングデータ数：" + str(y_test.shape))
y_test = y_test.flatten() # model入力用に変換
y_test[y_test==-1000]=0
#y_test = y_test[::1]
#print(y_test.shape)

np.save('X_test30_id' + '.npy', X_test)
np.save('y_test30_id' + '.npy', y_test)
del X_test,y_test


