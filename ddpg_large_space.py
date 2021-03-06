import copy
import random
import collections

import gym
import numpy as np
import chainer
import cupy
from chainer import functions as F
from chainer import links as L
from chainer import optimizers

import matplotlib.pyplot as plt


# ====================
# Hyperparameters
# ====================
CAPACITY = 100000
M = 1000
BATCH_SIZE = 64
GPU = 0 # cpu -1 , gpu id
TAU = 1e-2
#REWARD_SCALE= 1e-2
REWARD_SCALE=1
RENDER = True
GAMMA = 0.99
REPLAY_START_SIZE=500
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
# ===================
# Utilities
# ===================
def soft_update(src, dst, tau):
    for s, d in zip(src.params(), dst.params()):
        d.data[:] = tau*s.data + (1-tau)*d.data

# ===================
# Feature Extractor
# ===================
"""
class FeatureExtractor(chainer.Chain):
    def __init__(self):
        c1 = 16
        c2 = 32
        c3 = 32
        
        super(FeatureExtractor, self).__init__(
            conv0=L.Convolution2D(STATE_LENGTH, c1, 3, stride=1, pad=0),
            conv1=L.Convolution2D(c1, c2, 3, stride=1, pad=0),
            conv2=L.Convolution2D(c2, c3, 3, stride=1, pad=0),
            bnorm0=L.BatchNormalization(c1),
            bnorm1=L.BatchNormalization(c2),
            bnorm2=L.BatchNormalization(c3),
            )

    def __call__(self, s):
        s = s/255.
        h = F.max_pooling_2d(F.relu(self.bnorm0(self.conv0(s))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bnorm1(self.conv1(h))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bnorm2(self.conv2(h))), 2, stride=2)
        return h
"""
# ===================
# Critic
# ===================
class QNet(chainer.Chain):
    def __init__(self, s_dim, a_dim):

        initializer = chainer.initializers.HeNormal()
        c1 = 16
        c2 = 32
        c3 = 32
        fc_unit=250
        #self.fe = FeatureExtractor()
        #if GPU >= 0:
        #    self.fe.to_gpu()

        super(QNet, self).__init__(
            conv0=L.Convolution2D(STATE_LENGTH, c1, 3, stride=1, pad=0),
            conv1=L.Convolution2D(c1, c2, 3, stride=1, pad=0),
            conv2=L.Convolution2D(c2, c3, 3, stride=1, pad=0),
            fc0=L.Linear(None, fc_unit, initialW=initializer),
            fc1=L.Linear(fc_unit, 64, initialW=initializer),
            fc2=L.Linear(64, 1, initialW=initializer),
            bnorm0=L.BatchNormalization(c1),
            bnorm1=L.BatchNormalization(c2),
            bnorm2=L.BatchNormalization(c3),
            )
    def __call__(self, s, a):

        s = s/255.
        h = F.max_pooling_2d(F.relu(self.bnorm0(self.conv0(s))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bnorm1(self.conv1(h))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bnorm2(self.conv2(h))), 2, stride=2)
        #h = self.fe(s)
        h = F.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
        h = F.concat( (h, a), axis=1 )
        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        y = self.fc2(h)

        return y

    #def feature(self, s):
    #    return self.fe(s)

    #def predict(self, f, a):
    #    h = F.reshape(f, (f.shape[0], f.shape[1]*f.shape[2]*f.shape[3]))
    #    h = F.concat( (h, a), axis=1 )
    #    h = F.relu(self.fc0(h))
    #    y = self.fc1(h)
    #    return y

class Critic:
    def __init__(self, s_dim, a_dim=1):
        self.Q = QNet(s_dim, a_dim)

        if GPU >= 0:
            self.Q.to_gpu()

        self.Q_ = copy.deepcopy(self.Q)
        
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.Q)

    def update(self, q, t):
        loss = F.mean_squared_error(q, t)
        self.Q.cleargrads()
        loss.backward()
        self.optimizer.update()

    def target_update(self):
        soft_update(self.Q, self.Q_, TAU)

    def predict(self, s, a):
        with chainer.no_backprop_mode():
            q = self.Q(s, a).data
        return q      
        #return chainer.cuda.to_cpu(q)

    def target_predict(self, s, a):
        with chainer.no_backprop_mode():
            q = self.Q_(s, a).data
            #q = self.Q_.predict(s, a).data
        return chainer.cuda.to_cpu(q)


    def feature(self, s):
        with chainer.no_backprop_mode():
            f = self.Q.feature(s).data
        return f

# ===================
# Actor
# ===================
class Policy(chainer.Chain):
    def __init__(self, s_dim, a_num, a_low, a_high):

        self.low = a_low
        self.high = a_high
        self.a_dim = 1
        
        initializer = chainer.initializers.HeNormal()
        c1 = 16
        c2 = 32
        c3 = 32
        fc_unit = 250

        super(Policy, self).__init__(
            conv0=L.Convolution2D(STATE_LENGTH, c1, 3, stride=1, pad=0),
            conv1=L.Convolution2D(c1, c2, 3, stride=1, pad=0),
            conv2=L.Convolution2D(c2, c3, 3, stride=1, pad=0),
            fc0=L.Linear(None, fc_unit, initialW=initializer),
            fc1=L.Linear(fc_unit, 64, initialW=initializer),
            fc2=L.Linear(64, self.a_dim, initialW=initializer),
            bnorm0=L.BatchNormalization(c1),
            bnorm1=L.BatchNormalization(c2),
            bnorm2=L.BatchNormalization(c3)
            )

    def __call__(self, x):
        x = x/255.
        h = F.max_pooling_2d(F.relu(self.bnorm0(self.conv0(x))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bnorm1(self.conv1(h))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bnorm2(self.conv2(h))), 2, stride=2)

        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        y = self.fc2(h)
        return self.squash(y, 
                           self.xp.asarray(self.high),
                           self.xp.asarray(self.low))

    def squash(self, x, high, low):
        center = (high + low) / 2.
        scale = (high - low) / 2.

        return F.tanh(x)*scale + center

class Actor:
    def __init__(self, s_dim, a_num, a_low, a_high):
        self.policy = Policy(s_dim, a_num, a_low, a_high)
        
        if GPU >= 0:
            self.policy.to_gpu()

        self.policy_ = copy.deepcopy(self.policy)

        self.optimizer = optimizers.Adam(alpha=1e-4)
        self.optimizer.setup(self.policy)

    def predict(self, states):
        with chainer.no_backprop_mode():
            a = self.policy(states).data

        return chainer.cuda.to_cpu(a)

    def target_predict(self, states):

        with chainer.no_backprop_mode():
            a = self.policy_(states).data
        return chainer.cuda.to_cpu(a)

    def update(self, Q, s, n):
        loss = -F.sum(
            Q(s, self.policy(s) )
            ) / n

        self.policy.cleargrads()
        loss.backward()

        self.optimizer.update()

    def target_update(self):
        soft_update(self.policy, self.policy_, TAU)

# ====================
# Agent
# ====================
K=3
class Agent:
    def __init__(self, s_dim, a_num, a_low, a_high):
        self.low = a_low
        self.high = a_high
        self.s_dim = s_dim
        self.actor = Actor(s_dim, a_num, a_low, a_high)
        self.critic = Critic(s_dim, 1) # TODO action dim
        self.D = ReplayMemory(CAPACITY)
      
    def observe(self, transition):
        self.D.add(transition)
    
    def update(self):
        # samole from Replay Mem.
        mini_batch = self.D.sample(BATCH_SIZE)
        n = len(mini_batch)
        
        xp = self.critic.Q.xp

        s = xp.asarray([_[0] for _ in mini_batch], dtype=np.float32)
        
        a = xp.asarray([_[1] for _ in mini_batch], dtype=np.float32)
        a = xp.expand_dims(a, axis=1)
        r = xp.asarray([_[2] for _ in mini_batch], dtype=np.float32)
        s_ = xp.asarray([_[3] for _ in mini_batch], dtype=np.float32)
        done = xp.asarray([_[4] for _ in mini_batch], dtype=np.float32)
        
        q = F.reshape(
            self.critic.Q(s, a), (n,)
            ) 

        with chainer.no_backprop_mode():
            actions = xp.empty((n,1), dtype=np.float32)
            proto_a = self.actor.target_predict(s_)


            a_neighbors = xp.expand_dims(
                xp.asarray(
                [ self.neighbors(proto_a[i])for i in range(n)],
                dtype=np.float32
                ).T,
                axis=2)

            
            #print a_neighbors
            #f = self.critic.feature(s_)
            q_val = np.empty((K,n), dtype=np.float32)
            for i in xrange(K):
                q_val[i] = self.critic.target_predict(
                    #f,
                    s_,
                    a_neighbors[i]
                    ).reshape(n)
            q_val = q_val.T
            a_neighbors = a_neighbors.T.reshape(n, K, 1)
            for i in range(n):
                acts = a_neighbors[i]
                actions[i][0] = acts[ np.argmax(q_val[i]) ][0]
            """
            Mainly reduce computation cost
            a_neighbors = xp.expand_dims(
                xp.asarray(
                [ self.neighbors(proto_a[i])[j] for i in range(n) for j in range(K) ],
                dtype=np.float32
                ),
                axis=1
                )
            f = self.critic.feature(s_)
            features = xp.asarray(
                [f[i] for i in range(n) for j in range(K)],
                dtype=np.float32)
            q_val = self.critic.target_predict(
                features,
                a_neighbors)

            for i in range(n):
                offset = i*K
                acts = a_neighbors[offset:offset+K]
                q_vals = q_val[offset:offset+K]
                actions[i][0] = acts[ np.argmax(q_vals) ][0]
             
            """

            """
            Naive
            #a_neighbors = xp.asarray( 
            #    [ xp.expand_dims(self.neighbors(proto_a[i]), axis=1) for i in range(n)],
            #    dtype=np.float32)
            #states = xp.asarray(
            #    [s_[i] for i in range(n) for j in range(K)],
            #    dtype=np.float32)
            for i in range(n):
                a_neighbors = self.neighbors(proto_a[i])
            
                states = xp.asarray(
                    [s_[i] for _ in range(len(a_neighbors))], 
                    dtype=np.float32)

                q_val = self.critic.target_predict(
                    states, 
                    xp.expand_dims(a_neighbors.astype(np.float32), axis=1)
                    )

                actions[i][0] = a_neighbors[np.argmax(q_val)]
            """

            q_ = F.reshape(
            self.critic.Q_(s_, actions),
            (n,)
            )
            t = r + GAMMA*(1-done)*q_


        self.critic.update(q, t)
        self.actor.update(self.critic.Q, s, n)
        
        self.critic.target_update()
        self.actor.target_update()


    def neighbors(self, a, k=1):
        neighbor = np.empty( (K,), dtype=np.int32)

        tmp = a.astype(np.int32)[0]

        if tmp <= self.low:
            tmp = self.low
            neighbor[1] = tmp+1
            neighbor[2] = tmp+2
        elif tmp >= self.high:
            tmp = self.high
            neighbor[1] = tmp-1
            neighbor[2] = tmp-2
        else:
            neighbor[1] = tmp+1
            neighbor[2] = tmp-1
        

        neighbor[0] = tmp

        xp = self.critic.Q.xp
        #print neighbor
        return xp.asarray(neighbor, dtype=np.int32)

# ====================
# Replay Mem
# ====================
class ReplayMemory:
    def __init__(self, capacity, seed=123):
        self.buf = collections.deque(maxlen=capacity)
        random.seed(seed)

    # add (s, a, r, s', done) into replay buf 
    def add(self, transition):
        self.buf.append(transition)
        
    # sample n experiences from replay buf randomly
    def sample(self, n): # n = batch size
        n = min(n, len(self.buf))
        return random.sample(self.buf, n)

    def len(self):
        return len(self.buf)

# ====================
# Preprocessor
# ====================

from skimage.color import rgb2gray
from skimage.transform import resize
class Preprocessor:
    def __init__(self):
        self.state = None
    def init_state(self, obs):
        processed_obs = self._preprocess_observation(obs)
        state = [processed_obs for _ in xrange(STATE_LENGTH)]
        self.state = np.stack(state, axis=0)
    def obs2state(self, obs):
        processed_obs = self._preprocess_observation(obs)
        self.state = np.concatenate((
                self.state[1:, :, :],
                processed_obs[np.newaxis]),
                                    axis=0)
        return self.state
    def _preprocess_observation(self, obs):
        return np.asarray(resize(rgb2gray(obs), (FRAME_WIDTH, FRAME_HEIGHT))*255,
                          dtype=np.uint8)
        


# ====================
# Main loop
# ====================
def noise():
    return np.random.normal(scale=1.0)

MIN_EPSILON=0.01
STEPS_TO_DECAY_EPSILON=500000
def main():
    fig, ax = plt.subplots(1, 1)

    env = gym.make('Pong-v0')
    s_dim = env.observation_space.low.size
    #a_dim = env.action_space.size
    #a_high = env.action_space.high
    #a_low = env.action_space.low
    a_low = 0
    a_high = 5
    
    a_num = env.action_space.n
    print ("DEBUG %d %d")%(s_dim, a_num)

    preprocessor = Preprocessor()

    agent = Agent(s_dim, a_num, a_low, a_high)
    xp = agent.critic.Q.xp

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()

    Rs = []
    
    step = 0
    for episode in range(M):
        obs = env.reset()

        preprocessor.init_state(obs)

        done = False
        R = 0.0
        t = 0

        s = preprocessor.state

        while not done and t < env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'):
            state = xp.expand_dims(
                xp.asarray(s, dtype=np.float32),
                axis=0)

            # get next action
            #proto_a = agent.actor.predict(state) + noise() # proto_a is on cpu
            proto_a = agent.actor.predict(state)
            neighbors = agent.neighbors(proto_a) # neighbors on gpu
            ss = xp.asarray(
                [s for _ in xrange(K)],
                dtype=np.float32)

            q_val = agent.critic.predict(
                ss,
                xp.expand_dims(neighbors.astype(np.float32),
                               axis=1)                
                )

            a = chainer.cuda.to_cpu(neighbors[int(xp.argmax(q_val))])
            
            epsilon = 1.0 if agent.D.len() < REPLAY_START_SIZE else \
                max(MIN_EPSILON, np.interp(
                    step, [0, STEPS_TO_DECAY_EPSILON], [1.0, MIN_EPSILON]))
            
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            # execute action
            obs_, r, done, info = env.step(a)  
            s_ = preprocessor.obs2state(obs_)
            
            if RENDER:
                env.render()

            R += r

            agent.observe( (s, a, r*REWARD_SCALE, s_, done) )
            if len(agent.D.buf) >= REPLAY_START_SIZE:
                agent.update()

            s = s_

            t += 1
            step += 1
            #print step
        print("Total Reward:", R)
        
        Rs.append(R)

        ax.clear()
        ax.plot(Rs)
        fig.canvas.draw()
        plt.show()

if __name__ == "__main__":
    main()
