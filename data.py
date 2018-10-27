import numpy as np
import tensorflow as tf
import os,time
import warp

HOME = os.getenv("HOME")

# load data (one thread)
def load(test=False):
	path = 'data2'
	TrainImg = 255-np.load("{0}/image_train.npy".format(path))
	return {'TrainImg':TrainImg[:-1],'StandardImg':TrainImg[-1]}
'''
def load(test=False):
	path = 'sketchdata'
	TrainImg = 255-np.load("{0}/train_img_bell.npy".format(path))
	stdImg = 255-np.load("{0}/train_img_knife.npy".format(path))
	return {'TrainImg':stdImg,'StandardImg':TrainImg}
'''
'''
# make training batch
def makeBatch(batchSize,data,PH):
	N = len(data["TrainImg"])
	randIdx0 = np.random.randint(N,size=[batchSize])
	# put data in placeholders
	StandardImg = data['StandardImg']
	StandardImg = np.tile(StandardImg,[batchSize,1,1,1])
	[StandardData,WarpdData] = PH
	batch = {
		StandardData: StandardImg/255.0,
		WarpdData: data["TrainImg"][randIdx0]/255.0
	}
	shape = batch[WarpdData].shape
	appendarr = np.ones([shape[0],shape[1],shape[2],1])
	batch[WarpdData] = np.concatenate((batch[WarpdData],appendarr),axis = 3)
	return batch
'''
def makeBatch(batchSize,data,PH):
	N = len(data["TrainImg"])
	randIdx0 = np.random.randint(N,size=[batchSize])
	randIdx1 = np.random.randint(N,size=[batchSize])
	# put data in placeholders
	StandardImg = data['StandardImg']
	StandardImg = np.tile(StandardImg,[batchSize,1,1,1])
	[StandardData,WarpdData] = PH
	batch = {
		StandardData: StandardImg[randIdx0]/255.0,
		WarpdData: data["TrainImg"][randIdx1]/255.0
	}
	return batch

def makeBatch_test(batchSize,data):
	N = len(data["TrainImg"])
	randIdx0 = np.random.randint(N,size=[batchSize])

	StandardImg = data['StandardImg']
	StandardImg = np.tile(StandardImg,[batchSize,1,1,1])
	batch = {
		'StandardData': StandardImg/255.0,
		'WarpdData': data["TrainImg"][randIdx0]/255.0
	}
	return batch

# make test batch
def makeBatchEval(batchSize,testImage,PH):
	idxG = np.arange(batchSize)
	# put data in placeholders
	[image] = PH
	batch = {
		image: np.tile(testImage,[batchSize,1,1,1]),
	}
	shape = batch[image].shape
	appendarr = np.ones([shape[0],shape[1],shape[2],1])
	batch[image] = np.concatenate((batch[image],appendarr),axis = 3)
	return batch

# make test batch
def makeBatchEval_tps(batchSize,testImage,PH):
	idxG = np.arange(batchSize)
	# put data in placeholders
	[image] = PH
	batch = {
		image: np.tile(testImage,[batchSize,1,1,1]),
	}
	return batch