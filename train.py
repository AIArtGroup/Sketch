import numpy as np
import time
import util

import tensorflow as tf
import data
import graph,ThinPlateSpline
import scipy.misc
#probably don't change
GPUdevice = "/gpu:1"
dataH = 256
dataW = 256
stack_num = 1
#warpDim = 8
warpN = 5
#batch size
batchSize = 1
#perturbation
pert = 0.1

#run training to iteration number
toIt = 8000



#learning rate 
lrGP = 1e-5
lrGPdecay = 1.0
lrGPstep = 5000
#initialization stddev of graph
stdGP = 0.01


print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device(GPUdevice):
	# ------ define input data ------
	StandardData = tf.placeholder(tf.float32,shape=[batchSize,dataH,dataW,3])
	WarpdData = tf.placeholder(tf.float32,shape=[batchSize,dataH,dataW,3])
	PH = [StandardData,WarpdData]
	# ------ generate perturbation ------
	#StandardData = data.perturbBG(opt,StandardData)
	pPertFG = tf.zeros([batchSize,8])
	#geometric = graph_for_tps.PT_STN
	# ------ geometric predictor ------
	#image = geometric(WarpdData,pPertFG,stdGP,warpN,batchSize,dataH,dataW)
	
	# ------ define GP------
	geometric = graph.combine
	# ------ geometric predictor ------
	imageWarped = geometric(WarpdData,stdGP,batchSize,dataH,dataW,pPertFG,warpN)
	#(WarpdData,stdGP,batchSize,dataH,dataW)
	#(WarpdData,stdGP,batchSize,dataH,dataW,pPertFG,warpN)
	#(image,stdGP,batchSize,dataH,dataW,p,warpN)
	#(image,p,stdGP,warpN,batchSize,dataH,dataW)
	loss_GP = tf.reduce_sum(tf.abs(StandardData-imageWarped))
	tf.summary.scalar("loss__",loss_GP)

	# ------ optimizer ------
	vars_pt = [v for v in tf.global_variables() if "pt" in v.name]
	vars_tps = [v for v in tf.global_variables() if "tps0" in v.name]
	lrGP_PH = tf.placeholder(tf.float32,shape=[])
	vars_all = vars_pt+vars_tps
	with tf.name_scope("adam1"):
		optimGP1 = tf.train.AdamOptimizer(learning_rate=lrGP_PH).minimize(loss_GP,var_list=vars_pt)
	with tf.name_scope("adam2"):
		optimGP2 = tf.train.AdamOptimizer(learning_rate=lrGP_PH).minimize(loss_GP,var_list=vars_tps)
	with tf.name_scope("adam3"):
		optimGP3 = tf.train.AdamOptimizer(learning_rate=lrGP_PH).minimize(loss_GP,var_list=vars_all)


# load data
print(util.toMagenta("loading training data..."))
trainData = data.load()


# prepare model saver/summary writer
saver_GP = tf.train.Saver(max_to_keep=20)
summaryWriter = tf.summary.FileWriter("summary")
summary_op=tf.summary.merge_all()


print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True

with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	
	summaryWriter.add_graph(sess.graph)

	# training loop
	for i in range(toIt):
		lrGP = lrGP*lrGPdecay**(i//lrGPstep)
		batch = data.makeBatch(batchSize,trainData,PH)
		batch[lrGP_PH] = lrGP
		if(i<1200):
			optim = optimGP1
		else:
			optim = optimGP2
		'''
		if(i<2000):
			optim = optimGP1
		elif(i<10000):
			optim = optimGP2
		else:
			optim = optimGP3
		'''
		runList = [optim,loss_GP,vars_all,summary_op,imageWarped,StandardData]
		_,lg,var,summary_res,image,stimage = sess.run(runList,feed_dict=batch)
		summaryWriter.add_summary(summary_res,i)
		
		if (i+1)%10==0:
			print("it.{0}/{1}  lr={3}  loss={4}(GP) time={2}"
				.format(util.toCyan("{0}".format((i+1))),
						toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toYellow("{0:.0e}".format(lrGP)),
						util.toRed("{0:.4f}".format(lg)),
						))
		if  (i+1)%5000==0:
			saver_GP.save(sess,"model_0/models_it{0}_stack{1}.ckpt".format(i+1,stack_num))
	#print(image.shape,stimage.shape)
	#scipy.misc.imshow(1-image[0][0])
	#scipy.misc.imshow(1-stimage[0])
	saver_GP.save(sess,"model_0/models_it{0}_stack{1}.ckpt".format(toIt,stack_num))
print(util.toYellow("======= TRAINING DONE ======="))