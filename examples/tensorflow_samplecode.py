import tensorflow as tf
import tensorboard as tb

if __name__ == '__main__':

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # create symbolic variable
    # 'None' means that dimension can be of any length
    input = tf.placeholder(tf.float32, [None, 784])
    # Variable is modifiable tensor that lives in
    # TensorFlow's graph of interacting operations
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # Model's implementation:
    # tf.matmul(input, W) multiplies input by W
    output = tf.nn.softmax(tf.matmul(input, W) + b)

    # new placeholder for correct output
    label = tf.placeholder(tf.float32, [None, 10])

    # Implementation of the cross-entropy function:
    #
    # tf.log computes the logarithm of each element of output
    #
    # Next, we multiply each element of label with
    # the corresponding element of tf.log(output)
    #
    # tf.reduce_sum adds the elements in the second dimension of output,
    # due to the reduction_indices=[1] parameter
    # (sum over elements of the sample not over batch elements)
    #
    # tf.reduce_mean computes the mean over all the examples in the batch
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(output), reduction_indices=[1]))

    # Choose the optimization algorithm
    # minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # TensorFlow actually adds new operations to the graph which implement backpropagation and
    # gradient descent. Then it gives us back a single operation which, when run,
    # does a step of gradient descent training.

    # launch the model in an InteractiveSession:
    sess = tf.InteractiveSession()

    # create an operation to initialize the variables we created:
    tf.global_variables_initializer().run()

    # run the training step 1000 times:
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # run train_step feeding in the batches data to replace the placeholders:
        sess.run(train_step, feed_dict={input: batch_xs, label: batch_ys})

    # Evaluation of the model
    # Check if prediction corresponds to ground truth:
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))

    # To determine what fraction are correct,
    # we cast to floating point numbers and then take the mean:
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Finally, evaluate accuracu on testing data:
    print('Accuracy: %.2f%%' % (100 * sess.run(accuracy, feed_dict={input: mnist.test.images, label: mnist.test.labels})))