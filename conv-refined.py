import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

RESTORE_CHECKPOINT=False

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
x_ = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
is_training = tf.placeholder(tf.bool)

with tf.name_scope("conv1"):
    # W is the filter for the convolution, parameters are: [4,4] (filter sizee), 1 (input size), 4 (output size
    W1 = tf.Variable(tf.truncated_normal([5,5,1,4], 0, 1), name="W1")
    b1 = tf.Variable(tf.constant(0.1, tf.float32, [4]), name="b1")
    l1 = tf.nn.lrn(tf.nn.relu( tf.nn.conv2d(x_, W1, [1,1,1,1], "SAME") + b1 ))

with tf.name_scope("conv2"):
    W2 = tf.Variable(tf.truncated_normal([5,5,4,8], 0, 1), name="W2")
    b2 = tf.Variable(tf.constant(0.1, tf.float32, [8]), name="b2")
    l2 = tf.nn.lrn(tf.nn.relu( tf.nn.conv2d(l1, W2, [1,2,2,1], "SAME") + b2))

with tf.name_scope("conv3"):
    W3 = tf.Variable(tf.truncated_normal([4,4,8,12], 0, 1), name="W3")
    b3 = tf.Variable(tf.constant(0.1, tf.float32, [12]), name="b3")
    l3 = tf.nn.lrn(tf.nn.relu( tf.nn.conv2d(l2, W3, [1,2,2,1], "SAME") + b3))

with tf.name_scope("dense1"):
    l3_ = tf.reshape(l3, [-1, 7*7*12])
    W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], 0, 1), name="W4")
    b4 = tf.Variable(tf.constant(0.1, tf.float32, [200]), name="b4")
    l4 = tf.nn.relu( tf.matmul(l3_, W4) + b4 )
    l4 = tf.layers.dropout(l4, rate=0.5, training=is_training)

with tf.name_scope("out"):
    W5 = tf.Variable(tf.truncated_normal([200,10], 0, 1), name="W5")
    b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name="b5")
    l5 = tf.matmul(l4, W5) + b5

out = tf.nn.softmax(l5)

with tf.name_scope("loss"):
    loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=l5))
    train_loss_stat = tf.summary.scalar("Train loss", loss)
    test_loss_stat = tf.summary.scalar("Test loss", loss)

with tf.name_scope("accuracy"):
    correct_predictions = tf.cast(tf.equal(tf.argmax(out,1), tf.argmax(y,1)), "float")
    class_accuracy_op = tf.cast(tf.reduce_mean(correct_predictions), "float")
    train_accuracy_stat = tf.summary.scalar("Train accuracy", class_accuracy_op)
    test_accuracy_stat = tf.summary.scalar("Test accuracy", class_accuracy_op)

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

train_stats = tf.summary.merge([train_loss_stat, train_accuracy_stat])
test_stats = tf.summary.merge([test_loss_stat, test_accuracy_stat])

# Training

batch_size = 100
epochs = 200
num_step = 0
last_epoch = 0
ckpt_path = 'models/conv.ckpt'

init = tf.global_variables_initializer()
session = tf.Session()

saver = tf.train.Saver()

if RESTORE_CHECKPOINT and tf.train.checkpoint_exists(ckpt_path):
    saver.restore(session, ckpt_path)
else:
    session.run(init)


log_dir = time.strftime("logs/conv/%Y-%m-%d-%H:%M:%S", time.gmtime())
summary_writer = tf.summary.FileWriter(log_dir, session.graph)

mnist = input_data.read_data_sets("data/", one_hot=True)

while mnist.train.epochs_completed < epochs:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    session.run(train_step, feed_dict={x: batch_xs, y:batch_ys, is_training:True})
    num_step += 1

    if num_step % 200 == 0:
        tb_test = session.run(test_stats, feed_dict={x: mnist.test.images, y: mnist.test.labels, is_training:False})
        summary_writer.add_summary(tb_test, global_step=num_step)

        tb_train = session.run(train_stats, feed_dict={x:mnist.train.images, y:mnist.train.labels, is_training:False})
        summary_writer.add_summary(tb_train, global_step=num_step)


    if last_epoch != mnist.train.epochs_completed:
        last_epoch = mnist.train.epochs_completed

        saver.save(session, ckpt_path)

        train_loss, train_accuracy = session.run([loss, class_accuracy_op], feed_dict={x: mnist.train.images, y: mnist.train.labels, is_training:False})
        test_loss, test_accuracy = session.run([loss, class_accuracy_op], feed_dict={x: mnist.test.images, y: mnist.test.labels, is_training:False})

        print("epoch: %3d\ttrain accuracy: %2.5f\ttest accuracy: %2.5f\ttrain loss: %2.5f\ttest loss: %2.5f" % (last_epoch, train_accuracy, test_accuracy, train_loss, test_loss))
