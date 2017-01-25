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
REWARD_SCALE= 1e-3
RENDER = True
GAMMA = 0.99
REPLAY_START_SIZE=500

# ===================
# Utilities
# ===================
def soft_update(src, dst, tau):
    for s, d in zip(src.params(), dst.params()):
        d.data[:] = tau*s.data + (1-tau)*d.data

# ===================
# Critic
# ===================
class QNet(chainer.Chain):
    def __init__(self, s_dim, a_dim):
        super(QNet, self).__init__(
            l0 = L.Linear(s_dim + a_dim, 100),
            l1 = L.Linear(100, 50),
            l2 = L.Linear(50, 1, wscale=1e-3),
            )
    def __call__(self, s, a):
        x = F.concat( (s, a), axis=1 )
        h = F.relu( self.l0(x)  )
        h = F.relu( self.l1(h)  )
        return self.l2(h)        

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
        return chainer.cuda.to_cpu(q)

    def target_predict(self, s, a):
        with chainer.no_backprop_mode():
            q = self.Q_(s, a).data
        return chainer.cuda.to_cpu(q)

# ===================
# Actor
# ===================
class Policy(chainer.Chain):
    def __init__(self, s_dim, a_num, a_low, a_high):

        self.low = a_low
        self.high = a_high

        initializer = chainer.initializers.HeNormal()
        super(Policy, self).__init__(
            l0 = L.Linear(s_dim, 64),
            l1 = L.Linear(64, 1, initialW=initializer),
            )

    def __call__(self, x):
        h = F.relu( self.l0(x) )
        h = self.l1(h)
        return self.squash(h, 
                           self.xp.asarray(self.high),
                           self.xp.asarray(self.low))

    def squash(self, x, high, low):
        center = (high + low) / 2
        scale = (high - low) / 2
        return x*scale + center

class Actor:
    def __init__(self, s_dim, a_num, a_low, a_high):
        self.policy = Policy(s_dim, a_num, a_low, a_high)
        
        if GPU >= 0:
            self.policy.to_gpu()

        self.policy_ = copy.deepcopy(self.policy)

        self.optimizer = optimizers.Adam(alpha=1e-4)
        self.optimizer.setup(self.policy)

    def predict(self, states):
        xp = self.policy.xp
        states = xp.expand_dims(
            xp.asarray(states, dtype=np.float32), axis=0)
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
class Agent:
    def __init__(self, s_dim, a_num, a_low, a_high):
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

            #a_neighbors = xp.asarray( 
            #    [ xp.expand_dims(self.neighbors(proto_a[i]), axis=1) for i in range(n)],
            #    dtype=np.float32)
            a_neighbors = xp.expand_dims(
                xp.asarray(
                [ self.neighbors(proto_a[i])[j] for i in range(n) for j in range(2) ],
                dtype=np.float32
                ),
                axis=1
                )
            states = xp.asarray(
                [s_[i] for i in range(n) for j in range(2)],
                dtype=np.float32)
            
            q_val = self.critic.target_predict(
                states,
                a_neighbors)
            
            for i in range(n):
                offset = i*2
                acts = a_neighbors[offset:offset+2]
                q_vals = q_val[offset:offset+2]
                actions[i][0] = acts[ np.argmax(q_vals) ][0]
            
            """
            Naive
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
        xp = self.critic.Q.xp
        return xp.asarray([0,1], dtype=np.int32)

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



# ====================
# Main loop
# ====================
def noise():
    return np.random.normal(scale=0.4)

def main():
    fig, ax = plt.subplots(1, 1)

    env = gym.make('CartPole-v0')
    s_dim = env.observation_space.low.size
    #a_dim = env.action_space.size
    #a_high = env.action_space.high
    #a_low = env.action_space.low
    a_low = 0
    a_high = 1
    
    a_num = env.action_space.n
    print ("DEBUG %d %d")%(s_dim, a_num)

    agent = Agent(s_dim, a_num, a_low, a_high)

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()

    Rs = []

    for episode in range(M):
        s = env.reset()
        done = False
        R = 0.0
        t = 0

        while not done and t < env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'):
            
            # get next action
            proto_a = agent.actor.predict(s) + noise() # proto_a is on cpu
            neighbors = agent.neighbors(proto_a) # neighbors on gpu
        
            xp = agent.critic.Q.xp
            states = xp.asarray(
                [s for _ in range(len(neighbors))], 
                dtype=np.float32)
                                
            q_val = agent.critic.predict(
                    states,
                    xp.expand_dims(neighbors.astype(np.float32),
                                  axis=1)
                )

            a = chainer.cuda.to_cpu(neighbors[np.argmax(q_val)])

            # execute action
            s_, r, done, info = env.step(a)
            
            if RENDER:
                env.render()

            R += r

            agent.observe( (s, a, r*REWARD_SCALE, s_, done) )
            if len(agent.D.buf) >= REPLAY_START_SIZE:
                agent.update()

            s = s_

            t += 1
        
        print("Total Reward:", R)
        
        Rs.append(R)

        ax.clear()
        ax.plot(Rs)
        fig.canvas.draw()
        plt.show()

if __name__ == "__main__":
    main()
