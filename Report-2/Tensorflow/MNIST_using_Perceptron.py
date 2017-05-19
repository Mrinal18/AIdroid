import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = 'True')
def neural_network(X):

	nodel1 = 200
	nodel2 = 100
	nodel3 = 60
	nodel4 = 30
	output_node = 10

	layer1 = {'weights': tf.Variable(tf.zeros([784, nodel1])), 'biases': tf.Variable(tf.zeros([nodel1])) }
	layer2 = {'weights': tf.Variable(tf.zeros([nodel1, nodel2])), 'biases': tf.Variable(tf.zeros([nodel2])) }
	layer3 = {'weights': tf.Variable(tf.zeros([nodel2, nodel3])), 'biases': tf.Variable(tf.zeros([nodel3])) }
	layer4 = {'weights': tf.Variable(tf.zeros([nodel3, nodel4])), 'biases': tf.Variable(tf.zeros([nodel4])) }
	output = {'weights': tf.Variable(tf.zeros([nodel4, output_node])), 'biases': tf.Variable(tf.zeros([output_node])) }

	l1 = tf.add(tf.matmul(X, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
	l3 = tf.nn.sigmoid(l3)

	l4 = tf.add(tf.matmul(l3, layer4['weights']), layer4['biases'])
	l4 = tf.nn.sigmoid(l4)

	output = tf.add(tf.matmul(l4, output['weights']), output['biases'])
	output = tf.nn.softmax(output)

	return output

def training(X):
	
	y = neural_network(X)

	y_ = tf.placeholder(tf.float32, [None, 10])

	cost = -tf.reduce_sum(y_*tf.log(y))	

	optimizer = tf.train.GradientDescentOptimizer(0.001)

	train_sets = optimizer.minimize(cost)

	correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	batch_size = 100

	init = tf.global_variables_initializer()

	with tf.Session() as sess:

		sess.run(init)

		for i in range(10000):

			batch_x, batch_y = mnist.train.next_batch(100)
			train_data = {X: batch_x, y_ : batch_y}

			sess.run(train_sets, feed_dict = train_data)

			a, c = sess.run([accuracy, correct], feed_dict = train_data)
	print a

X = tf.placeholder(tf.float32, [None, 784])

training(X)





