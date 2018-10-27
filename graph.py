import numpy as np
import tensorflow as tf
import ThinPlateSpline,itertools
import warp

# build geometric predictor
def TPS_STN(image,stdGP,batchSize,dataH,dataW,stack_num=0):
	shape = image.shape
	#print('input shape  {0}'.format(image.shape))
	with tf.variable_scope("tps{0}".format(stack_num)):
		source_control_points = generate_source_control_point(batchSize)
		target_control_points = source_control_points
		warpDim = 2*source_control_points.shape[1]
		imageConcat = image
		feat = image
		#print(feat.shape)
		with tf.variable_scope("conv1"): feat,imageConcat = conv2Layer(feat,imageConcat,32,stdGP) # 72x72
		with tf.variable_scope("conv2"): feat,imageConcat = conv2Layer(feat,imageConcat,64,stdGP) # 36x36
		with tf.variable_scope("conv3"): feat,imageConcat = conv2Layer(feat,imageConcat,128,stdGP) # 18x18
		with tf.variable_scope("conv4"): feat,imageConcat = conv2Layer(feat,imageConcat,256,stdGP) # 9x9
		with tf.variable_scope("conv5"): feat,imageConcat = conv2Layer(feat,imageConcat,512,stdGP) # 5x5
		feat = tf.reshape(feat,[batchSize,-1])
		with tf.variable_scope("fc6"): feat = linearLayer(feat,256,stdGP)
		with tf.variable_scope("fc7"): feat = linearLayer(feat,warpDim,stdGP,final=True)
		delta = tf.reshape(feat,[batchSize,-1,2])
		image = tf.expand_dims(image,axis =-1)
		target_control_points += delta
		#target_control_points = tf.expand_dims(target_control_points,axis=0)
		#source_control_points = tf.expand_dims(source_control_points,axis=0)
		#print(image[0].shape, source_control_points[0].shape, target_control_points[0].shape)
		WarpImageAll = []
		#print(image.shape)
		for i in range(batchSize):
			WarpImage = ThinPlateSpline.ThinPlateSpline(tf.expand_dims(image[i],axis=0), tf.expand_dims(source_control_points[i],axis=0), tf.expand_dims(target_control_points[i],axis=0), [dataH,dataW,3])
			WarpImageAll.append(WarpImage[0])
		WarpImageAll = tf.convert_to_tensor(WarpImageAll)
		WarpImageAll = tf.reshape(WarpImageAll,shape)
		#print('output shape  {0}'.format(WarpImageAll.shape))
	return WarpImageAll

def PT_STN(image,p,stdGP,warpN,batchSize,dataH,dataW):
	shape = image.shape
	appendarr = tf.ones([shape[0],shape[1],shape[2],1])
	image = tf.concat((image,appendarr),axis = 3)
	with tf.variable_scope("pt"):
		dp = None
		# define recurrent spatial transformations
		for l in range(warpN):
			with tf.variable_scope("warp{0}".format(l)):
				pMtrx = warp.vec2mtrx(batchSize,p)
				imagewarp = warp.transformImage(batchSize,image,pMtrx,dataH,dataW)
				# geometric predictor
				imageConcat = imagewarp
				feat = imageConcat 
				with tf.variable_scope("conv1"): feat,imageConcat = conv2Layer(feat,imageConcat,32,stdGP) # 72x72
				with tf.variable_scope("conv2"): feat,imageConcat = conv2Layer(feat,imageConcat,64,stdGP) # 36x36
				with tf.variable_scope("conv3"): feat,imageConcat = conv2Layer(feat,imageConcat,128,stdGP) # 18x18
				with tf.variable_scope("conv4"): feat,imageConcat = conv2Layer(feat,imageConcat,256,stdGP) # 9x9
				with tf.variable_scope("conv5"): feat,imageConcat = conv2Layer(feat,imageConcat,512,stdGP) # 5x5
				feat = tf.reshape(feat,[batchSize,-1])
				with tf.variable_scope("fc6"): feat = linearLayer(feat,256,stdGP)
				with tf.variable_scope("fc7"): feat = linearLayer(feat,8,stdGP,final=True)
				dp = feat
				p = warp.compose(p,dp)
		# warp image with final p
		#print(imagewarp.shape)
		pMtrx = warp.vec2mtrx(batchSize,p)
		imagewarp = warp.transformImage(batchSize,imagewarp,pMtrx,dataH,dataW)
		colorFG,maskFG = imagewarp[:,:,:,:3],imagewarp[:,:,:,3:]
		imagewarp = colorFG*maskFG
		#print(imagewarp.shape)
	return imagewarp

def combine(image,stdGP,batchSize,dataH,dataW,p,warpN):
	with tf.variable_scope('combine'):
		image = PT_STN(image,p,stdGP,warpN,batchSize,dataH,dataW)
		image = TPS_STN(image,stdGP,batchSize,dataH,dataW)
	return image

def stacked_STN(image,stdGP,batchSize,dataH,dataW,stack_num=1):
	with tf.variable_scope("stacked_STN",reuse = False):
		for i in range(stack_num):
			image = TPS_STN(image,stdGP,batchSize,dataH,dataW,i)
	return image
	
# auxiliary function for creating weight and bias
def createVariable(weightShape,biasShape=None,stddev=None):
	if biasShape is None: biasShape = [weightShape[-1]]
	weight = tf.get_variable("weight",shape=weightShape,dtype=tf.float32,
									  initializer=tf.random_normal_initializer(stddev=stddev))
	bias = tf.get_variable("bias",shape=biasShape,dtype=tf.float32,
								  initializer=tf.constant_initializer(0.0))
	return weight,bias

def generate_source_control_point(batchSize):
	r1 = 0.9
	r2 = 0.9
	grid_height = 15
	grid_width  = 15
	ini_control_point = np.array(list(itertools.product(
		np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (grid_height - 1)),
		np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (grid_width - 1)),)))
	source_control_points = tf.convert_to_tensor(ini_control_point)
	source_control_points = tf.expand_dims(source_control_points,axis=0)
	source_control_points = tf.tile(source_control_points,[batchSize,1,1])
	return tf.cast(source_control_points,tf.float32)

def downsample(x):
	padH,padW = int(x.shape[1])%2,int(x.shape[2])%2
	if padH!=0 or padW!=0: x = tf.pad(x,[[0,0],[0,padH],[0,padW],[0,0]])
	return tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],"VALID")
	
def conv2Layer(feat,imageConcat,outDim,stdGP,final=False):
	weight,bias = createVariable([4,4,(feat.shape[-1]),outDim],stddev=stdGP)
	conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
	feat = tf.nn.relu(conv)
	imageConcat = downsample(imageConcat)
	feat = tf.concat([feat,imageConcat],axis=3)
	return (feat if not final else conv),imageConcat

def linearLayer(feat,outDim,stdGP,final=False):
	weight,bias = createVariable([(feat.shape[-1]),outDim],stddev=stdGP)
	fc = tf.matmul(feat,weight)+bias
	feat = tf.nn.relu(fc)
	return feat if not final else fc