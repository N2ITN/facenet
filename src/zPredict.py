from tensorflow.python.framework import ops            
import tensorflow as tf
import tensorflow.contrib.slim as slim 
from tensorflow.python.platform import gfile
import pandas as pd
import numpy as np
import importlib
import que_es
import random
import facenet
import Image


def load_img_files():
    with open('/home/zach/repos/facenet/files_list','r') as temp:
        raw =  [x.replace(' \n','').split() for x in temp.readlines()]
        random.shuffle(raw)
        files = [r[0] for r in raw]
        targets =[r[1] for r in raw]
        return files, targets
def convert(files):
    for f in files:
        print f
        current = Image.open(f)
        image_size = current.size[0]
        file_contents = tf.read_file(f)
        name = f.rsplit('/')[-2]
        image = tf.image.decode_png(file_contents)#, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_whitening(image)
        image = tf.expand_dims(image, 0, name = name)
        yield image
graph_path = "/home/zach/repos/facenet/zGraph/z_trained.pb"

def predict(image):
    with tf.Session() as sess:    
        with gfile.FastGFile(graph_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')                  
        resnet = importlib.import_module('models.nn4', 'inference')
        weight_decay = 0.
        keep_probability = 1.
        # images = convert(files)
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        prelogits, _ = resnet.inference(image, keep_probability,
                    phase_train=True, weight_decay=weight_decay)
        logits = slim.fully_connected(prelogits, 7, activation_fn=None,  
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),  
            weights_regularizer=slim.l2_regularizer(0.), 
            scope='Logits', reuse=False)

        # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        labels = ops.convert_to_tensor([i for i in range(0,7)], dtype=tf.int32)
        
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        

        feed_dict =  { 'input:0': image.eval(), phase_train_placeholder:False}
        # emb, lab = sess.run([embeddings, labels])
        # emb_array = np.zeros((7, int(embeddings.get_shape()[1])))
        # nums = ops.convert_to_tensor(emb_array, dtype=tf.float32)
        # emb_array[lab] = emb
        emb, lab = sess.run([embeddings, labels])
        '''
        emb_array = np.zeros((7, int(embeddings.get_shape()[1])))
        
        emb_array[lab] = emb
        '''
        embits = slim.fully_connected(emb, 7, activation_fn=None,  
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),  
            weights_regularizer=slim.l2_regularizer(0.), 
            scope='Embits', reuse=False)
        # print dir(embits)

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        print image.name[:-2]
        target_arr = [0]*7
        target = int(image.name[:1])    
        target_arr[target] = 1
        print image.name
        print 'target ',   target, target_arr
        outDict = {'target': target}
        def extract_values(tens,name):
            # print tens.eval()[0]
            # print logits.eval()[0]
            
            probLog = tf.nn.softmax(tens)
            # probEmb = tf.nn.softmax(embits)
            
            classLog =  probLog.eval(feed_dict=feed_dict)
            # classEmb =  probEmb.eval(feed_dict=feed_dict)
                

            from operator import add
            arrLog = [0]*7
            # arrEmb = [0]*7
            
            # for i, n in enumerate(classEmb):
                # arrEmb = map(add, arrEmb, n)
            
            # arrEmb = [(x*100)/sum(arrEmb) for x in arrEmb]
            for i,n in enumerate(classLog):
                arrLog = map(add, arrLog, n)
            arrLog = [int((x*100)/sum(arrLog)) for x in arrLog]
            # result_arr =  map(add, arrLog, arrEmb)

            # maxRes = pd.Series(result_arr).idxmax()
            # em = pd.Series(arrEmb).idxmax()

            lo = res = pd.Series(arrLog).idxmax()
            
             
            # print 'log+emb:', res, result_arr
            print name, lo, arrLog 
            return [name,lo]
            # print 'embeds :', em, arrEmb
            
            # out =  {'target':target, 'Average':res, 'Embeddings':em, 'Logits':lo}
            # out =  {'target':target, name:lo}
        lgts = extract_values(logits,'logits')
        outDict[lgts[0]]=lgts[1]
        mbts = extract_values(embits,'embits')
        outDict[mbts[0]]=mbts[1]
        with open('test_results.txt', 'a') as myfile:
            myfile.write(str(outDict)+ ', ')

        
        return outDict

            
def testr(i):
    f = files
    for im in xrange(i):
        
        image = next(convert(f))

        
        # print image
        # print predict(image)
        print predict(image)
    # for j in range(num):
    #     im = next(test_imgs)
    #     print predict(im)
        # tf.reset_default_graph()


# test_imgs = convert(files)

# print list(testr(5))[0]
files,targets = load_img_files()
def main():

    testr(1)
main()
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

# cor = np.rot90(next((predict(15))))
''''
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
       emb_array[lab] = emb
        # print emb_array

        # with tf.variable_scope('Logits'):
        #     slim = tf.contrib.slim
        #     n = int(prelogits.get_shape()[1])
        #     m =  7
        #     w = tf.get_variable('w', shape=[n,m], dtype=tf.float32, 
        #         initializer=tf.truncated_normal_initializer(stddev=0.1), 
        #         regularizer=slim.l2_regularizer(weight_decay),
        #         trainable=False)
        #     b = tf.get_variable('b', [m], initializer=None, trainable=False)
        #     # print embeddings
        #     # print emb_array
        #     xxx= ops.convert_to_tensor(emb_array, dtype=tf.float32)
        #     # print xxx
            
        #     embits = tf.matmul(xxx, w) + b
        #     logits = tf.matmul(prelogits, w) + b
        # probEmb = tf.nn.softmax(embits)
        # probLog = tf.nn.softmax(logits)
        # # print prob

        # feed_dict =  { 'input:0': image.eval(), phase_train_placeholder:False} 
        # classEmb =  probEmb.eval(feed_dict=feed_dict)
        # classLog =  probLog.eval(feed_dict=feed_dict)




'''


