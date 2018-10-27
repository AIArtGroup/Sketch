import numpy as np
import util
import scipy.misc
import tensorflow as tf
import data
import graph,ThinPlateSpline


GPUdevice = "/gpu:1"
batchSize = 1 
dataH = 256
dataW = 256
warpDim = 8
warpN = 5
#perturbation
pert = 0.
#initialization stddev of graph
stdGP = 0.01
#number of spatial transformations residue
#warpN = 5 

#test image
loadImage = 'data2/starfish.png'
#model name to load
stack_num = 1
trained_model = "model_0/models_it8000_stack{0}.ckpt".format(stack_num)



print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device(GPUdevice):
	# ------ define input data ------
    WarpdData = tf.placeholder(tf.float32,shape=[batchSize,dataH,dataW,3])
    PH = [WarpdData]
    pPertFG = pert*tf.random_normal([batchSize,warpDim])
    # ------ define GP ------
    geometric = graph.combine
    # ------ geometric predictor ------
    imageWarped = geometric(WarpdData,stdGP,batchSize,dataH,dataW,pPertFG,warpN)
    # ------ optimizer ------
    #varsGP = [v for v in tf.global_variables() if "geometric" in v.name]

# prepare model saver/summary writer
saver_GP = tf.train.Saver()

print(util.toYellow("======= EVALUATION START ======="))
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
    sess.run(tf.global_variables_initializer())
	#restore the model
    saver_GP.restore(sess,trained_model)
    print(util.toMagenta("start evaluation..."))

    testImage = util.imread(loadImage)
    batch = data.makeBatchEval_tps(batchSize,testImage,PH)
    runList = [WarpdData,imageWarped]
    ic0,icf = sess.run(runList,feed_dict=batch)
    #print(ic0.shape,icf.shape)
    util.imsave("eval/image__input.png",1-ic0[0])
    util.imsave("eval/image__output{0}.png".format(stack_num),1-icf[0])

print(util.toYellow("======= EVALUATION DONE ======="))
