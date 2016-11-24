from tensorflow.python.platform import gfile
import pandas as pd
import numpy as np
import importlib
import que_es
import random
import facenet
# tf.import_graph_def(graph_def)
# y = tf.nn.softmax(logits)


def load_img_files():
    with open('/home/zach/repos/facenet/files_list','r') as temp:
        raw =  [x.replace(' \n','').split() for x in temp.readlines()]
        random.shuffle(raw)
        files = [r[0] for r in raw]
        targets =[r[1] for r in raw]
        return files, targets
def convert(files):
    for f in files:
        image_size = 160
        file_contents = tf.read_file(f)
        name = f.rsplit('/')[-2]
        image = tf.image.decode_jpeg(file_contents, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_whitening(image)
        image = tf.expand_dims(image, 0, name = name)
        yield image
import tensorflow as tf
graph_path = "/home/zach/repos/facenet/zGraph/frozen_big.pb"
# model_dir = '/home/zach/repos/facenet/20161104-005712'
# meta_file = 'model-20161104-005712.meta'
# ckpt_file = 'model-20161104-005712.ckpt-12000'
# facenet.load_model(model_dir, meta_file, ckpt_file) 

def predict(image):
    
    with gfile.FastGFile(graph_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')                  
    resnet = importlib.import_module('models.nn4', 'inference')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')           
    weight_decay = 0.
    keep_probability = 1.
    images = convert(files)
    # phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    

    prelogits, _ = resnet.inference(image, keep_probability,
                    phase_train=True, weight_decay=weight_decay)
    print prelogits
    from tensorflow.python.framework import ops            
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    labels = ops.convert_to_tensor([0,1,2,3,4,5,6,7,8], dtype=tf.int32)
    sess.run(tf.initialize_all_variables())
    emb, lab = sess.run([embeddings, labels])
    emb_array = np.zeros((9, int(embeddings.get_shape()[1])))
    # print emb, lab
    emb_array[lab] = emb
    # print emb_array


    with tf.variable_scope('Logits'):
        slim = tf.contrib.slim
        n = int(prelogits.get_shape()[1])
        m =  7
        w = tf.get_variable('w', shape=[n,m], dtype=tf.float32, 
            initializer=tf.truncated_normal_initializer(stddev=0.1), 
            regularizer=slim.l2_regularizer(weight_decay),
            trainable=False)
        b = tf.get_variable('b', [m], initializer=None, trainable=False)
        # print embeddings
        # print emb_array
        xxx= ops.convert_to_tensor(emb_array, dtype=tf.float32)
        # print xxx
        
        embits = tf.matmul(xxx, w) + b
        logits = tf.matmul(prelogits, w) + b
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    

    probEmb = tf.nn.softmax(embits)
    probLog = tf.nn.softmax(logits)
    # print prob
    target = int(image.name[:1])
    feed_dict =  { 'input:0': image.eval(), phase_train_placeholder:False} 
    classEmb =  probEmb.eval(feed_dict=feed_dict)
    classLog =  probLog.eval(feed_dict=feed_dict)
    # classification = sess.run(prob, feed_dict=feed_dict)
    from operator import add
    arrEmb = [0]*7
    arrLog = [0]*7
    target_arr = [0]*7
    for i, n in enumerate(classEmb):
        arrEmb = map(add, arrEmb, n)
    arrEmb = [int((x*100)/sum(arrEmb)) for x in arrEmb]
    for i,n in enumerate(classLog):
        arrLog = map(add, arrLog, n)
    arrLog = [int((x*100)/sum(arrLog)) for x in arrLog]
    result_arr =  map(add, arrLog, arrEmb)            
    res = pd.Series(result_arr).idxmax()
    target_arr[target] = 1
    
    
    # top_r = [round(float(x*10)/sum(class2[-1]),1) for x in class2[-1]]
    # bottom_r = [round(float(x*10)/sum(class2[0]),1) for x in class2[0]]
    print image.name
    print 'target:', target_arr
    print 'all___:', result_arr
    # print 'top___:', top_r
    # print 'bottom:', bottom_r
    
    print target, res#, top, bottom

    with open('test_results_500.txt', 'a') as myfile:
        myfile.write(str([target, res])+ ', ')
        return [target, res]  
        
def testr(i):
    f = files
    for im in xrange(i):
    
        image = next(convert(f))
        print image
        # print image
        # print predict(image)
        print predict(image)
    # for j in range(num):
    #     im = next(test_imgs)
    #     print predict(im)
        # tf.reset_default_graph()

files,targets = load_img_files()
# test_imgs = convert(files)

# print list(testr(5))[0]
with tf.Session() as sess:
    testr(2)

# str(next(predict(500)))
# with open('test_results_500.txt', 'a') as myfile:
#     rline = 
#     myfile.write(rline)
#     print rline
    

# nd = np.asarray(ans)
# df = pd.DataFrame(ans)
# print nd
# cor = nd.corr()
# print cor
exit()
# cor = np.rot90(next((predict(15))))

print cor
exit() 

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")

# Load the datset of correlations between cortical brain networks
df = sns.load_dataset([targets, results])
corrmat = df.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)

# Use matplotlib directly to emphasize known networks

f.tight_layout()

sns.plt.show()


            

            # ############################################
            # from tensorflow.python.framework import ops
            # # Node for input images
            # eval_image_batch = images
            # eval_image_batch = tf.identity(eval_image_batch, name='input')
            # # eval_prelogits, _ = resnet.inference(eval_image_batch, 1.0, 
            # #     phase_train=False, weight_decay=0.0, reuse=True)
            # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            
            # emb_array = np.zeros((1, int(prelogits.get_shape()[1])))
            # sess.run(tf.initialize_all_variables())
            # sess.run(tf.initialize_local_variables())
            # labels = ops.convert_to_tensor([0,1,2,3,4,5,6], dtype=tf.int32)

            # emb, lab = tf.matmul(embeddings, labels)
            # # emb, lab = sess.run([embeddings, labels])
            # emb_array[lab] = emb
            
            # import lfw   
            # _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, 666, 1, nrof_folds=1)
            #######################################################
