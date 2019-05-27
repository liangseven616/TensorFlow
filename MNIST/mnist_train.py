from  tensorflow.examples.tutorials.mnist import  input_data
import  tensorflow as tf

mnist = input_data.read_data_sets("./MNIST_data",one_hot=1)
batch_size = 100
n_batch = mnist.train.num_examples//batch_size


x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

weight = tf.Variable(tf.zeros([784,10]))
biase = tf.Variable(tf.zeros([10]))

y_ = tf.nn.softmax(tf.matmul(x,weight)+biase)
loss = tf.reduce_mean(tf.square(y-y_))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

currect_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(currect_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epcoh in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epcoh)+",Test Accuracy"+str(acc))
