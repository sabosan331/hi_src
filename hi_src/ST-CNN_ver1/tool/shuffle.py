import numpy as np
dataPath = '/media/aoki/559a0841-86d1-4a13-bad8-8cab1f416f10/kojima/data/samp10_2/'
savePath = '/media/aoki/559a0841-86d1-4a13-bad8-8cab1f416f10/kojima/data/shuffle1228/' 
sc_h = 90
sc_w = 120
depth =16
X1_train = np.empty( (0,depth,sc_h,sc_w),float)
X2_train = np.empty( (0,depth,sc_h,sc_w),float)
y_train = np.empty(0,int)

dataSt=0
dataEnd =4
dataNum = dataEnd - dataSt
for i in range(dataSt,dataEnd):
	print("------------------load: " + str(i) + "------------------------" )
	X1_load = np.load(dataPath+'/GS/X_train10_id' + str(i) + '.npy')
	X2_load = np.load(dataPath+'/depth/X_trainDP10_id' + str(i) + '.npy')
	y_load = np.load(dataPath+'/GS/y_train10_id' + str(i) + '.npy')

	# X1_load = np.load(savePath+'/GS/X_train10_id' + str(i) + '.npy')
	# X2_load = np.load(savePath+'/depth/X_trainDP10_id' + str(i) + '.npy')
	# y_load = np.load(savePath+'/GS/y_train10_id' + str(i) + '.npy')

	X1_train = np.append(X1_train,X1_load,axis=0)
	X2_train = np.append(X2_train,X2_load,axis=0)	
	y_train = np.append(y_train,y_load,axis=0)

	print(X1_train.shape)
	print(X2_train.shape)
	print(y_train.shape)


print("------------------shuffle: " + str(i) + "------------------------" )

data_size = y_train.shape[0]
# data shuffle
shuffle_indices = np.random.permutation(np.arange(data_size))
X1_train = X1_train[shuffle_indices]
X2_train = X2_train[shuffle_indices]
y_train = y_train[shuffle_indices]

print(X1_train.shape)
print(X2_train.shape)
print(y_train.shape)


splitLen = int(data_size/dataNum)

idx = 0
for i in range(dataSt,dataEnd):
	print("-------------------- save: " + str(i) + "------------------------" )
	frameSt = idx * splitLen
	frameEnd = (idx+1) * splitLen
	idx = idx +1
	print("start:"+ str(frameSt) + "end:"+ str(frameEnd) )
	X1_tmp = X1_train[frameSt:frameEnd:1]
	X2_tmp = X2_train[frameSt:frameEnd:1]
	y_tmp = y_train[frameSt:frameEnd:1]
	print(X1_tmp.shape)
	print(X2_tmp.shape)
	print(y_tmp.shape)
	np.save(savePath + '/GS/X_train10_id' + str(i) + '.npy', X1_tmp)
	np.save(savePath + '/depth/X_trainDP10_id' + str(i) + '.npy', X2_tmp)
	np.save(savePath + '/GS/y_train10_id' + str(i) + '.npy', y_tmp)
