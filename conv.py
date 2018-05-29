import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
x_ = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")

with tf.name_scope("conv1"):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4]), name="W1")
    b1 = tf.Variable(tf.constant(0.1, tf.float32, [4]), name="b1")
    l1 = tf.nn.lrn(tf.nn.relu(tf.nn.conv2d(x_, W1, [1,1,1,1], "SAME") + b1))

with tf.name_scope("conv2"):
    W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8]), name="W2")
    b2 = tf.Variable(tf.constant(0.1, tf.float32, [8]), name="b2")
    l2 = tf.nn.lrn(tf.nn.relu(tf.nn.conv2d(l1, W2, [1,2,2,1], "SAME") + b2))

with tf.name_scope("conv3"):
    W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12]), name="W3")
    b3 = tf.Variable(tf.constant(0.1, tf.float32, [12]), name="b3")
    l3 = tf.nn.lrn(tf.nn.relu(tf.nn.conv2d(l2, W3, [1,2,2,1], "SAME") + b3))

with tf.name_scope("dense1"):
    l3_ = tf.reshape(l3, [-1, 7 * 7 * 12])
    W4 = tf.Variable(tf.truncated_normal([7*7*12, 200]), name="W4")
    b4 = tf.Variable(tf.constant(0.1, tf.float32, [200]), name="b4")
    l4 = tf.nn.relu(tf.matmul(l3_, W4) + b4)

with tf.name_scope("out"):
    W5 = tf.Variable(tf.truncated_normal([200,10]), name="W5")
    b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name="b5")
    l5 = tf.matmul(l4, W5) + b5

out = tf.nn.softmax(l5, name="output")

loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=l5) )

train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()


session.run(init)

mnist = input_data.read_data_sets("data/", one_hot=True)
   

while mnist.train.epochs_completed < 500:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run( train_step, feed_dict = { x:batch_xs, y: batch_ys } )
    print(session.run(loss, feed_dict = { x:batch_xs, y: batch_ys }))
