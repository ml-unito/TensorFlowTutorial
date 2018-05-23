import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

x = tf.placeholder( tf.float32, [None, 784])
y = tf.placeholder( tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 30], 0, 1))
b1 = tf.Variable(tf.random_normal([30], 0, 1))

l1 = tf.nn.sigmoid( tf.matmul(x, W1) + b1 )

W2 = tf.Variable(tf.random_normal([30, 10], 0 ,1))
b2 = tf.Variable(tf.random_normal([10], 0, 1))

l2 = tf.nn.sigmoid( tf.matmul(l1, W2) + b2 )
out = tf.nn.softmax(l2)

loss =  tf.reduce_mean(tf.reduce_sum(- y * tf.log(out), axis=1 ))
loss_stat = tf.summary.scalar("loss",loss)

accuracy = tf.cast(tf.equal(tf.argmax(out, axis=1), tf.argmax(y, axis=1)), tf.float32)
accuracy = tf.cast(tf.reduce_mean(accuracy), tf.float32)
accuracy_stat = tf.summary.scalar("accuracy",accuracy)

optimizer = tf.train.GradientDescentOptimizer(3.0)
train_step = optimizer.minimize(loss)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

session = tf.Session()
init = tf.global_variables_initializer()
model_saver = tf.train.Saver()


if tf.train.checkpoint_exists("models/mln"):
    model_saver.restore(session, "models/mln")
else:
    session.run(init)

num_epochs = 20
it_num = 0


saver = tf.summary.FileWriter('logs', session.graph)
all_stats = tf.summary.merge_all()

while mnist.train.epochs_completed  < 20:
    it_num += 1
    batch_xs, batch_ys = mnist.train.next_batch(10)
    session.run(train_step, feed_dict={x: batch_xs, y: batch_ys} )
    # loss_val = session.run(loss, {x: mnist.train.images, y: mnist.train.labels})

    if it_num % 20 == 0:
        stats = session.run(all_stats, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        saver.add_summary(stats, global_step=it_num)

    if it_num % 100 == 0:
        model_saver.save(session, "models/mln")
