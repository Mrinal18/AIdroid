import tensorflow as tf
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

n_pixels1 = 5
n_filters1 = 16

n_pixels2 = 5
n_filters2 = 36

n_classes = 10

img_size = 28
img_shape = (img_size, img_size)
n_channels = 1

img_size_flat = img_size * img_size

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))


def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_convlayer(input, n_inputch, filter_size, n_filters, use_pooling = True):
	
	shape = [filter_size, filter_size, n_inputch, n_filters]
	weights = new_weights(shape)
	biases = new_biases(length = n_filters)
	
	layer = tf.nn.conv2d(input = input, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')
	layer = layer + biases
	
	if use_pooling:
		layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	layer = tf.nn.relu(layer)
	return layer, weights

def flatten(layer):
	layer_shape = layer.get_shape()                    #layer_shape = [num_images, img_height, img_length, n_channel]
	
	n_features = layer_shape[1:4].num_elements()    #this does img_height*imglenth*n_channel

	layer_flat = tf.reshape(layer, [-1, n_features])    #this apparently makes the layer [num_images, n_features]

	return n_features, layer_flat

def new_fconlayer(input , n_inputs, n_outputs, use_relu = True):

	shape = [n_inputs, n_outputs]

	weights = new_weights(shape = shape)

	biases = new_biases(length = n_outputs)

	layer = tf.add(tf.matmul(input, weights), biases)

	if use_relu:

		layer = tf.nn.relu(layer)

	return layer

x = tf.placeholder(tf.float32, shape = [None, img_size_flat])
x_image = tf.reshape(x, [-1, img_size, img_size, n_channels])

y_true = tf.placeholder(tf.float32, [None, n_classes])
y_true_cls = tf.argmax(y_true, 1)

layer_conv1, weights_conv1 = new_convlayer(input = x_image, n_inputch = 1, filter_size = n_pixels1, n_filters = n_filters1, use_pooling = True)
layer_conv2, weights_conv2 = new_convlayer(input = layer_conv1, n_inputch = n_filters1, filter_size = n_pixels2, n_filters = n_filters2, use_pooling = True)

n_flat, flat = flatten(layer_conv2)

fc_layer1 = new_fconlayer(input = flat, n_inputs = n_flat, n_outputs = 128, use_relu = True)
fc_layer2 = new_fconlayer(input = fc_layer1, n_inputs = 128, n_outputs = 10, use_relu = False)

y_pred = tf.nn.softmax(fc_layer2)
y_pred_cls = tf.argmax(y_pred, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc_layer2, labels = y_true))
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train_sets = optimizer.minimize(cost)

correct = tf.equal(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
train_size_batch = 64
n_epochs = 10000

with tf.Session() as sess:
	sess.run(init)

	for i in range(n_epochs):
		
		epoch_x, epoch_y = mnist.train.next_batch(train_size_batch)
		train_data = sess.run(train_sets, feed_dict = {x: epoch_x, y_true : epoch_y})
		c, a = sess.run([correct, accuracy], feed_dict = {x: epoch_x, y_true: epoch_y})
		test_x, test_y = mnist.test.images, mnist.test.labels
		#sess.run(y_pred, feed_dict = {x: test_x, y_true : test_y})
		c1, a1 = sess.run([correct, accuracy], feed_dict = {x: test_x, y_true: test_y})
	print(a1)

sess.close()

