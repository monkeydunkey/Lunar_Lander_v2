from __future__ import print_function
import gym
import tensorflow as tf

import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

tf.set_random_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
#Log Directory for tensorboard
LOGDIR = "visualize/12"
updateTargetNetwork = 100


class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        tf.reset_default_graph()
        self.sess = tf.Session()

        # A few starter hyperparameters
        # hyperparameters
        self.gamma = 0.99
        self.h1 = 64
        self.h2 = 64
        self.h3 = 64
        self.l2_reg = 1e-6
        self.max_episode_step = 1000
        self.update_slow_target_every = 100
        self.batch_size = 1024
        self.eps_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_length = 1e5
        self.epsilon_decay_exp = 0.97
        self.num_episodes = 0
        self.num_steps = 0
        self.epsilon_linear_step = (self.eps_start-self.epsilon_end)/self.epsilon_decay_length
        # memory
        self.replay_memory = ReplayMemory(1e6)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 2000

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.target_input = tf.placeholder(dtype=tf.float32, shape=[None]+ list(self.env.observation_space.shape)) # input to slow target network

        with tf.variable_scope('q_network') as scope:
            self.q_values = self.build_model(self.observation_input)

        with tf.variable_scope('target_network') as scope:
            self.target_q_values = self.build_model(self.observation_input, False)

        self.q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
        self.q_target_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')

        # update values for slowly-changing target network to match current critic network
        update_slow_target_ops = []
        for i, slow_target_var in enumerate(self.q_target_network_vars):
            update_slow_target_op = slow_target_var.assign(self.q_network_vars[i])
            update_slow_target_ops.append(update_slow_target_op)

        self.update_slow_target_op = tf.group(*update_slow_target_ops, name='update_slow_target')

        # define your update operations here...
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.target = tf.placeholder(tf.float32, shape=[None])

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        #Calculating the action q value is taken from https://github.com/dennybritz/reinforcement-learning/tree/master/DQN
        gather_indices = tf.range(self.batch_size) * tf.shape(self.q_values)[1] + self.actions
        self.action_predictions = tf.gather(tf.reshape(self.q_values, [-1]), gather_indices)
        self.loss = tf.losses.huber_loss(self.target, self.action_predictions)#tf.squared_difference(self.target, self.action_predictions)

        #Adding a regularization term for the weights
        for var in self.q_network_vars:
            if not 'bias' in var.name:
                self.loss += self.l2_reg * 0.5 * tf.nn.l2_loss(var)
        #self.loss = (self.target-self.action_predictions)**2
        #self.losses = tf.reduce_mean(self.loss)
        self.minimizer = tf.train.AdamOptimizer(learning_rate = 1e-6).minimize(self.loss) #tf.train.GradientDescentOptimizer(1e-5).minimize(self.losses)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(LOGDIR)
        self.writer.add_graph(self.sess.graph)
        self.count = 0

        # Summaries for Tensorboard
        tf.summary.scalar("loss", self.loss)
        #tf.summary.scalar("loss_hist", self.losses),
        tf.summary.histogram("q_values_hist", self.q_values),
        tf.summary.scalar("max_q_value", tf.reduce_max(self.q_values))
        self.summ = tf.summary.merge_all()

    def build_model(self, observation_input, trainable=True, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values.
        Currently returns an op that gives all zeros.
        """
        hidden = tf.layers.dense(observation_input, self.h1, activation = tf.nn.relu, trainable = trainable, name = 'dense')
        hidden_2 = tf.layers.dense(hidden, self.h2, activation = tf.nn.relu, trainable = trainable, name = 'dense_1')
        hidden_3 = tf.layers.dense(hidden_2, self.h3, activation = tf.nn.relu, trainable = trainable, name = 'dense_2')
        action_values = tf.squeeze(tf.layers.dense(hidden_3, self.env.action_space.n, trainable = trainable, name = "qValueLayer"))
        return action_values

    def select_action(self, obs, evaluation_mode=False):
        if np.random.uniform(0,1) < self.eps_start and not evaluation_mode:
            finalAction=env.action_space.sample()
            return finalAction

        obs = np.reshape(obs, [1, self.env.observation_space.shape[0]])
        output=self.sess.run(self.q_values, feed_dict={self.observation_input:obs})
        finalAction=np.argmax(output)
        return finalAction

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """
        if self.replay_memory.__len__() < self.min_replay_size:
            return
        else:
            new_samples = self.replay_memory.sample(self.batch_size)
        obs,action,next_obs,reward,done = zip(*new_samples)
        targets = np.zeros(self.batch_size)
        nextObsQValue = self.sess.run(self.target_q_values, feed_dict={self.observation_input: np.array(next_obs)})
        for i, sample in enumerate(new_samples):
            #currentQValue[i] = qValues[i][sample.action]
            if sample.terminal:
                targets[i] = sample.reward
            else:
                targets[i] = np.max(nextObsQValue[i]) * self.gamma + sample.reward
        _, summaries = self.sess.run([self.minimizer, self.summ]
                       ,feed_dict={self.target:targets,self.observation_input:np.array(obs), self.actions: np.array(action)})
        self.writer.add_summary(summaries, self.count)
        self.count += 1
    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        for i in xrange(self.max_episode_step):
            if done:
                break
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            self.replay_memory.push(obs,action,next_obs,reward,done)
            obs = next_obs
            self.num_steps += 1
            self.update()
            #Start with an initial linear decay and after some time use exponential decay
            if self.num_steps < self.epsilon_decay_length:
                self.eps_start -= self.epsilon_linear_step
            elif done:
                self.eps_start = self.eps_start * self.epsilon_decay_exp

        self.num_episodes += 1
        #After every episode decreasing the epsilon
        if self.num_episodes % self.update_slow_target_every == 0:
            self.sess.run(self.update_slow_target_op)


    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False

        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ", total_reward)
        print (str(self.eps_start)+'-----'+str(self.num_episodes))
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

    def fc_layer(self, input, size_in, size_out, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out]), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[size_out]), name="B")
            act = tf.matmul(input, w) + b
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return act

def train(dqn):
    #estimator_copy = ModelParametersCopier(dqn.q_vals, dqn.targetNetwork)
    for i in count(1) :
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()

def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    for i in range(10):
        dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)
    print ('Evaluation Argument', args.eval)
    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
