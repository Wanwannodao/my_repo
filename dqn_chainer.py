import gym
import copy, random, math
import numpy as np
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

# Error Clipping
def hubert_loss(t, x): # sqrt(1+a^2)-1
    err = t-x
    return F.mean( F.sqrt(1 + F.square(err))-1, axis=-1) # inner axis

class ActionValue(Chain):
    def __init__ (self, stateCnt, actionCnt):
        # the top dimension = batch
        super(ActionValue, self).__init__(
            l1=L.Linear(stateCnt, 64),
            l2=L.Linear(64, actionCnt)
            )

    #def __call__(self, x, t):
    #    return F.huber_loss(x, t, delta=np.float32(1.0))

    def predict(self, states):
        h1 = F.relu(self.l1(states))
        return self.l2(h1)

class Policy(Chain):
    def __init__ (self, stateCnt, actionCnt):
        self.stateCnt = stateCnt

        self.model = ActionValue(stateCnt, actionCnt)
        self.model_ = copy.deepcopy(self.model) # target network

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.use_cleargrads()
        self.optimizer.setup(self.model)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False): # (sample=1, state=stateCnt)
        return self.predict(s.reshape(1, self.stateCnt), target=False)
        
    def train(self, x, y):
        self.model.zerograds()
        #loss = F.huber_loss(self.model.predict(Variable(x)), Variable(y), delta=np.float32(1.0))
        loss = F.mean_squared_error(self.model.predict(Variable(x)), Variable(y))
        loss.backward()

        self.optimizer.clip_grads(1)
        self.optimizer.update()


    def updateTargetModel(self):
        self.model_.copyparams(self.model)
        

#---------Replay Mem---------
class ReplayMemory:
    transitions = []
    def __init__ (self, capacity):
        self.capacity = capacity
    
    def push(self, transition):
        self.transitions.append(transition)
        
        if (len(self.transitions) > self.capacity):
            self.transitions.pop(0)
            
    def sample(self, n):
        n = min(n, len(self.transitions))
        return random.sample(self.transitions, n)

    def isFull(self):
        return len(self.transitions) >= self.capacity

#---------Agent--------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001


UPDATE_TARGET_FREQUENCY = 1000
class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.policy = Policy(stateCnt, actionCnt)
        self.memory = ReplayMemory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax( self.policy.predictOne(s.astype(np.float32)).data )

    def observe(self, transition):
        self.memory.push(transition)
        
        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.policy.updateTargetModel()
        
        #if self.steps % 100 == 0:
            
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        
    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[0] for o in batch], dtype=np.float32)
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch], dtype=np.float32)
        
        #x = self.policy.predict( Variable(states) )
        #q = x.data
        q = self.policy.predict( Variable(states) ).data
        q_ = self.policy.predict( Variable(states_) , target=True).data # prediciton form the target network

        x = np.zeros((batchLen, self.stateCnt), dtype=np.float32)
        y = np.zeros((batchLen, self.actionCnt), dtype=np.float32)

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = q[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA*np.amax(q_[i])
            
            x[i] = s
            y[i] = t

        self.policy.train(x, y)

class RandomAgent:
    memory = ReplayMemory(MEMORY_CAPACITY)
    def __init__(self, actionCnt): 
        self.actionCnt = actionCnt
    def act(self, s):
        return random.randint(0, self.actionCnt-1)
    def observe(self, transition):
        self.memory.push(transition)
    def replay(self):
        pass
                          
#---------main loop----------
def run(agent, env):
    s = env.reset() # initial_state
    R = 0 # total_reward

    while True:
        env.render()
        
        a = agent.act(s)
        s_, r, done, info = env.step(a)

        if done:
            s_ = None

        agent.observe( (s, a, r, s_) )
        agent.replay()

        s = s_
        R += r
        
        if done:
            break

    print("Total Reward:", R)
        
    
def main():
    env = gym.make('CartPole-v0')

    stateCnt = env.observation_space.shape[0]
    actionCnt = env.action_space.n
    
    agent = Agent(stateCnt, actionCnt)
    randomAgent = RandomAgent(actionCnt)

    try:
        while randomAgent.memory.isFull() == False:
            run(randomAgent, env)
        
        agent.memory.transitions = randomAgent.memory.transitions
        randomAgent = None
        
        while True:
            run(agent, env)
    finally:
        serializers.save_npz('network/model.model', agent.policy.model)

if __name__ == '__main__':
    main()
