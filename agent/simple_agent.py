import numpy as np
import random
import config

# import tensorflow as tf

# class DDPG(object):
#     def __init__(self, a_dim, a_bound, memory_length, TAU=0.01, LR_C=1e-4, LR_A=1e-4, batch_size=64):
#         self.memory_s = np.zeros((memory_length, 2,  config.Map.Height, config.Map.Width), dtype=np.float32)
#         self.memory_a = np.zeros((memory_length, a_dim, a_dim), dtype=np.float32)
#         self.memory_r = np.zeros((memory_length, 1), dtype=np.float32)
#         self.pointer = 0
#         self.memory_length = memory_length
#         self.batch_size = batch_size
#         self.sess = tf.Session()
#
#         self.a_dim, self.a_bound = a_dim, a_bound,
#         self.S = tf.placeholder(tf.float32, [None, config.Map.Height, config.Map.Width], 's')
#         self.S_ = tf.placeholder(tf.float32, [None, config.Map.Height, config.Map.Width], 's_')
#         self.R = tf.placeholder(tf.float32, [None, 1], 'r')
#
#         with tf.variable_scope('Actor'):
#             self.a = self._build_a(self.S, scope='eval', trainable=True)
#             a_ = self._build_a(self.S_, scope='target', trainable=False)
#         with tf.variable_scope('Critic'):
#             # assign self.a = a in memory when calculating q for td_error,
#             # otherwise the self.a is from Actor when updating Actor
#             q = self._build_c(self.S, self.a, scope='eval', trainable=True)
#             q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
#
#         # networks parameters
#         self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
#         self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
#         self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
#         self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
#
#         # target net replacement
#         self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
#                              for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
#
#         q_target = self.R
#         # in the feed_dic for the td_error, the self.a should change to actions in memory
#         td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
#         self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
#
#         a_loss = - tf.reduce_mean(q)    # maximize the q
#         self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
#
#         self.sess.run(tf.global_variables_initializer())
#
#     def choose_action(self, s):
#         return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
#
#     def learn(self):
#         # soft target replacement
#         self.sess.run(self.soft_replace)
#
#         indices = np.random.choice(self.memory_length, size=self.batch_size)
#         bt = self.memory[indices, :]
#         bs = bt[:, :self.s_dim]
#         ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
#         br = bt[:, -self.s_dim - 1: -self.s_dim]
#         bs_ = bt[:, -self.s_dim:]
#
#         self.sess.run(self.atrain, {self.S: bs})
#         self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
#
#     def store_transition(self, s, a, r, s_):
#         transition = np.hstack((s, a, [r], s_))
#         index = self.pointer % self.memory_length  # replace the old memory with new memory
#         self.memory[index, :] = transition
#         self.pointer += 1
#
#     def _build_a(self, s, scope, trainable):
#         with tf.variable_scope(scope):
#             net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
#             a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
#             return tf.multiply(a, self.a_bound, name='scaled_a')
#
#     def _build_c(self, s, a, scope, trainable):
#         with tf.variable_scope(scope):
#             n_l1 = 30
#             w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
#             w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
#             b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
#             net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
#             return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

class Agent():
    def __init__(self):
        np.random.seed(123123)
        self.env = None
        self.memory = []

        self.ob = None
        self.r = None

    def test(self, env):
        self.env = env
        observation = self.env.reset()
        done = False
        while not done:
            action = np.zeros((config.Game.AgentNum))
            ob, r, done, info = self.env.step(action)
            self.memory.append(sum(r))
        total_reward = sum(self.memory)
        self.memory = []
        return total_reward
