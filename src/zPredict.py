import que_es
import random

# tf.import_graph_def(graph_def)
# y = tf.nn.softmax(logits)


def load_img_files():
    with open('/home/zach/repos/facenet/files_list','r') as temp:
        
        raw =  [x.replace(' \n','').split() for x in temp.readlines()]
        random.shuffle(raw)
        files = [r[0] for r in raw]
        targets =[r[1] for r in raw]
        return files, targets

    
    
    

import tensorflow as tf
graph_path = "/home/zach/repos/facenet/zGraph/z_2.pb"

def predict():
    from tensorflow.python.platform import gfile
                    
    with tf.Session() as sess:
        # print("load graph")
        with gfile.FastGFile(graph_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')                  
            
            def convert(files):
                for f in files:
                    image_size = 224
                    file_contents = tf.read_file(f)
                    # print f
                    image =  tf.image.decode_jpeg(file_contents, channels=3)
                    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
                    image.set_shape((image_size, image_size, 3))
                    image = tf.image.per_image_whitening(image)
                    image = tf.expand_dims(image, 0, name = 'input')
                    yield image
       
            test_imgs = convert(files)
            import importlib
            resnet = importlib.import_module('models.nn4', 'inference')

            # images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
            # images = tf.placeholder(tf.bool,  name='phase_train')
            def pred_img(test_imgs):
                weight_decay = 2e-4
                keep_probability = 0.8
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                images =  next(test_imgs)
                print images
                # images = tf.identity(images, name='input')
                prelogits, _ = resnet.inference(images, keep_probability, 
                            phase_train=phase_train_placeholder, weight_decay=weight_decay)
                print "prelogits done"

                slim = tf.contrib.slim
                with tf.variable_scope('Logits'):
                    n = int(prelogits.get_shape()[1])
                    
                    m =  7
                    w = tf.get_variable('w', shape=[n,m], dtype=tf.float32, 
                        initializer=tf.truncated_normal_initializer(stddev=0.1), 
                        regularizer=slim.l2_regularizer(weight_decay),
                        trainable=True)
                    b = tf.get_variable('b', [m], initializer=None, trainable=True)
                    logits = tf.matmul(prelogits, w) + b 
                print "logits done"
                prob = tf.nn.softmax(logits)
                sess.run(tf.initialize_all_variables())
                sess.run(tf.initialize_local_variables())
                print "session started"
                
                for i, x in enumerate(test_imgs):
                    print i
                    im = next(test_imgs)
                    target = targets[i] 
                    feed_dict =  { 'input:0': im.eval(), phase_train_placeholder:True} 
                # class2 =  prob.eval(feed_dict=feed_dict)
                    classification = sess.run(prob, feed_dict=feed_dict)
                   
                    y = 0
                    trac = []
                    for i, n in enumerate(classification):
                        # if i > 20: return
                        z = 0
                        for ii, e in enumerate(n):
                            if e > z:
                                z = e
                                y = ii 
                        trac.append(y)
                    
                    res = max(set(trac), key=trac.count) + 1
                    print target, res
                    if  res == target:
                        yield True
                    else: yield False






            
            x = pred_img(test_imgs)
            print [y for y in x]
print "RETURN"

files,targets = load_img_files()
predict()
 



            
# persisted_result = sess.graph.get_tensor_by_name("InceptionResnetV1/Logits/Dropout/cond/pred_id:0")
# tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
# input_img = tf.identity(test_img, name='input')

'''
phase_train_placeholder = tf.placeholder(tf.bool,  name='phase_train')
for i, x in enumerate(files):
    t1 =  sess.run('InceptionResnetV1/Logits/Dropout/cond/pred_id:0', feed_dict={ phase_train_placeholder.eval(), next(test_imgs).eval()})
    print t1
    if i > 5: 
        exit() 
'''