import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# %matplotlib inline
import pickle
from numpy import array

dir_path = os.path.dirname(os.path.realpath(__file__))

# TRAIN_DIR = dir_path + '/input/train/'
TRAIN_DIR = dir_path + '/check_train/'
print(TRAIN_DIR)
TEST_DIR = dir_path+ '/check_test/'
print(TEST_DIR)


# On the kaggle notebook
# we only take the first 2000 from the training set
# and only the first 1000 from the test set
# REMOVE [0:2000] and [0:1000] when running locally
train_image_file_names = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)][0:2000] 
test_image_file_names = [TEST_DIR+i for i in os.listdir(TEST_DIR)][0:707]


# Slow, yet simple implementation with tensorflow
# could be rewritten to be much faster
# (which is not really needed as it takes less than 5 minutes on my laptop)
def decode_image(image_file_names, resize_func=None):
    
    images = []
    
    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file = tf.read_file(file_name)
        image = tf.image.decode_jpeg(file)
        if resize_func != None:
            image = resize_func(image)
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()   
        for i in range(len(image_file_names)):
            images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))
            if (i+1) % 1000 == 0:
                print('Images processed: ',i+1)
        
        session.close()
    
    return images



train_images = decode_image(train_image_file_names)
test_images = decode_image(test_image_file_names)
all_images = train_images + test_images


# Check mean aspect ratio (width/height), mean width and mean height
width = []
height = []
aspect_ratio = []
for image in all_images:
    h, w, d = np.shape(image)
    aspect_ratio.append(float(w) / float(h))
    width.append(w)
    height.append(h)


# Free up some memory
del train_images
del test_images
del all_images


# WIDTH=500
# HEIGHT=500
# WIDTH=256
# HEIGHT=256
WIDTH=128
HEIGHT=128
# WIDTH=64
# HEIGHT=64
resize_func = lambda image: tf.image.resize_image_with_crop_or_pad(image, HEIGHT, WIDTH)


processed_train_images = decode_image(train_image_file_names, resize_func=resize_func)
processed_test_images = decode_image(test_image_file_names, resize_func=resize_func)


# Chech the shapes
print(np.shape(processed_train_images))
print(np.shape(processed_test_images))

# processed_train_image = array(processed_train_images[0]).reshape(1, 500,500,3)

# Create one hot encoding for labels
labels = [[1., 0.] if 'dog' in name else [0., 1.] for name in train_image_file_names]

labels_tests = [[1., 0.] if 'dog' in name else [0., 1.] for name in test_image_file_names]

print(np.shape(labels))


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, shape=[None, 128 , 128 , 3]) # 
xs1 = tf.reshape(xs, shape=[-1, 128 * 128 * 3])
ys = tf.placeholder(tf.float32, shape=[None, 2])

# add output layer
prediction = add_layer(xs1,128*128*3 , 2,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    # images, labels = iterator.get_next()
    sess.run(train_step, feed_dict={xs: processed_train_images, ys: labels})
    y_pre = sess.run(prediction, feed_dict={xs: processed_train_images})
    if i % 50 == 0:
        print(compute_accuracy(processed_test_images, labels_tests))
    


# for item in processed_test_images:
#     item1 = tf.reshape(item, shape=[-1, 64 * 64 * 3])
#     y_pre = sess.run(prediction, feed_dict={xs: item1})
#     check = tf.argmax(y_pre,1)
#     print(check)