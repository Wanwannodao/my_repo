import numpy as np
import tensorflow as tf

def build_shared_network(X):
    conv1 = tf.contrib.layers.conv2d(
        X, 32, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
    conv2 = tf.contrib.layers.conv2d(
        conv1, 64, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

    fc1 = tf.contrib.layers.fully_connected(
        inputs=tf.contrib.layers.flatten(conv2),
        num_outputs=512,
        activation_fn=tf.nn.relu,
        scope="fc1")
    fc2 = tf.contrib.layers.fully_connected(
        inputs=fc1,
        num_outputs=128,
        scope="fc2")
    return fc2

class PolicyEstimator():
    def __init__(self, num_outputs, reuse=False, trainable=True):
        self.num_outputs = num_outputs

        self.states = tf.placeholder(shape=[None, 16, 64, 32], dtype=tf.uint8, name="X")
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.states) / 255.0
        batch_size = tf.shape(self.states)[0]

        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_network(X)

        with tf.variable_scope("policy_net"):
            self.logits = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits)

            self.predictions = {
                "logits": self.logits,
                "probs": self.probs
                }

            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            gather_indices = tf.range(batch_size)*tf.shape(self.probs)[1] + self.actions
            self.picked_actions_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.losses = - (tf.log(self.picked_actions_probs) * self.targets + 0.01*self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            if trainable:
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                               global_step=tf.contrib.framework.get_global_step())


class ValueEstimator():
    def __init__(self, reuse=False, trainable=True):
        self.states = tf.placeholder(shape=[None, 16, 64, 32], dtype=tf.uint8, name="X")
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        X = tf.to_float(self.states) / 255.0

        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_network(X)

        with tf.variable_scope("value_net"):
            self.logits = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None)
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")
            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = {
                "logits": self.logits
                }

            if trainable:
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                               global_step=tf.contrib.framework.get_global_step())
                

                                             
                                    
