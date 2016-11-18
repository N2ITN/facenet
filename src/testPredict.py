import que_es

import tensorflow as tf
graph_path = "/home/zach/repos/facenet/zGraph/z_2.pb"
# tf.import_graph_def(graph_def)
# y = tf.nn.softmax(logits)
with open('files_only','r') as temp:

    files =  [x.replace(' \n','') for x in  temp.readlines()]


### ENUMERATE
# for x, y in enumerate(files):
#     print x,y
#     if x > 100:
#         exit()
# exit()


img = "/home/zach/repos/facenet/obamaDisgust.jpg" 

# feed_dict = {x: img}
# classification = tf.run(y, feed_dict)
# print classification

# y.eval(feed_dict)


def stack():
    from tensorflow.python.platform import gfile
    with tf.Session() as persisted_sess:
        # print("load graph")
        with gfile.FastGFile(graph_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            # print("map variables")
            # print dir(graph_def.node[0])
            # print graph_def.node[0].op
            # print graph_def.node[0].name

            def node_query(n):
                test_node = graph_def.node[n]
                print
                # print dir(test_node)
                print
                print test_node.name, test_node.op
                print test_node.input
                print test_node.attr
                
                

            
            # [node_query(x) for x in xrange(5369,5398)]
            # exit()

            def explore_nodes(x=None):  
                with open('../nodes_.txt','w') as n:
                        
                    for m in graph_def.node:
                        n.write(str([str(m.name),  str(m.op)])+'\n')

                            
                        
                        
            # explore_nodes()
            # exit()

            # print ([x for x in dir(graph_def.ListFields) if '__' not in x])
            
            
            # predictions = tf.nn.softmax(logits)
            
            
            
            
            def convert(files):
                for f in files:
                    image_size = 224
                    file_contents = tf.read_file(f)
                    image =  tf.image.decode_jpeg(file_contents, channels=3)
                    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        
        
                    image.set_shape((image_size, image_size, 3))
                    image = tf.image.per_image_whitening(image)
                    image = tf.expand_dims(image, 0, name = 'pic')
                    yield image
                


            test_imgs = convert(files)
 
            
            persisted_result = persisted_sess.graph.get_tensor_by_name("InceptionResnetV1/Logits/Dropout/cond/pred_id:0")
            tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
            phase_train_placeholder = tf.placeholder(tf.bool,  name='phase_train')
            # input_img = tf.identity(test_img, name='input')
 
            for i, x in enumerate(files):
                t1 =  persisted_sess.run('InceptionResnetV1/Logits/Dropout/cond/pred_id:0', feed_dict={ phase_train_placeholder.eval(), next(test_imgs).eval()})
                print t1
                if i > 5: 
                    exit()
#tf.nn.softmax(logits)
# stack()
import importlib
resnet = importlib.import_module('models.nn4', 'inference')
images = tf.placeholder("float32", [None, 224, 224, 3], name="images")

weight_decay = 2e-4
keep_probability = 0.8
phase_train_placeholder = tf.placeholder(tf.bool,  name='phase_train')
prelogits, _ = resnet.inference(images, keep_probability, 
            phase_train=False, weight_decay=weight_decay)
slim = tf.contrib.slim
with tf.variable_scope('Logits'):
    n = int(prelogits.get_shape()[1])
    m = 0#len(train_set)
    w = tf.get_variable('w', shape=[n,m], dtype=tf.float32, 
        initializer=tf.truncated_normal_initializer(stddev=0.1), 
        regularizer=slim.l2_regularizer(weight_decay),
        trainable=True)
    b = tf.get_variable('b', [m], initializer=None, trainable=True)
    logits = tf.matmul(prelogits, w) + b 
# logits = resnet.inference(images, is_training=False, 
    # num_blocks=num_blocks, preprocess=True, bottleneck=True) 
prob = tf.nn.softmax(logits)
           # print dir(next(test_imgs))
            # print next(test_imgs).eval()

            # print next(test_imgs).get_shape()
            # exit()
#   pred = sess.run(predictions, feed_dict=feed_dict)          
            
            
            # test_imgs = tf.identity(test_imgs, name='input')
            # obama = cv2.imread(img)
            # que_es.esto(obama)
            # print 
            
            
            
            # print test_img
            # exit()
            
            # print test_img.eval.im_class
            
        
            
            # print test_img.dtype

            # print test_img.eval
            # print test_img.value_index
            # print test_img.op
            # print test_img.graph
            # print obama.flat
            # print obama.flatten
            # print obama.shape
            

            #print (convert(files[0]).eval)
            # im = cv2.imread("abc.tiff")