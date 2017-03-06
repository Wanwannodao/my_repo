import numpy as np
import subprocess

#P_FNAME="mem.log"
#T_FNAME="thread_map.csv"
#K_FNAME="test"
WARP_SIZE=32
np.set_printoptions(threshold=np.inf)

# ====================
# Environment
# ====================
class Environment:
    def __init__(self, thread_num, p_fname, t_fname, k_fname, comp=False):
        self.t_num = thread_num

        # initialize thread map
        self.t_map = np.asarray(
            [ _ for _ in range(thread_num) ],
            dtype=np.int32)

        # filename of profiled data
        self.p_fname = p_fname
        # filename of thread map (CSV)
        self.t_fname = t_fname
        # filename of executable file
        self.k_fname = k_fname

        self.actions_num = int(thread_num*(thread_num-1)/2)


        self.prev_transactions = 0

        if comp:
            self.compile()
        
    def load_profiled_data(self):
        data = np.loadtxt(self.p_fname, delimiter=" ")
        return data

    def reset(self):
        self.t_map = np.asarray(
            [ _ for _ in range(self.t_num) ],
            dtype=np.int32)
        np.savetxt(self.t_fname, self.t_map, fmt='%0.d')
        
        transactions = self.exec_kernel(0)
        self.prev_transactions = transactions
        obs = self.exec_kernel(1)
        state = self.addr2mem(obs)
        return state

    def preprocess(self, data):
        # sort with respect to tid
        data = data[ data[:,0].argsort() ]
        # extract addresses
        data = data[:,1]
        # subtract min address
        data = data - data.min()

        return data.reshape(len(data), 1) # 1 is size of refs

    def addr2mem(self, data):
        # sort with respect to tid
        data = data[ data[:,0].argsort() ]
        # extract addresses
        data = data[:,1]
        # subtract min address
        data = data - data.min()
        # divide by data size
        data = data/4


        #print ("DEBUG [{}, {}]".format(data.min(), data.max()))
        # TODO segment size / data size
        width = int(256 / 4) # scaled segment size w.r.t data size 
        height = int((data.max() - data.min() + 1)/width) # address range
        depth = int(len(data) / WARP_SIZE) # corresponding to warp axis

        #print ("DEBUG [{},{},{}]".format(width,height,depth))

        mem = np.zeros((height, width, depth), dtype=np.float16)

        # create binary image representing memory map
        for i in range(depth):
            offset = i*WARP_SIZE
            warp = data[offset:offset+WARP_SIZE]
            for j in range(WARP_SIZE):
                addr = warp[j]
                mem[int(addr/width)][int(addr%width)][i] = 255
        #print ("DEBUG mem")
        #print (data[0:32])
        #print (mem[0])

        return mem

    def swap_tid(self, x, y):
        t = self.t_map[x]
        self.t_map[x] = self.t_map[y]
        self.t_map[y] = t

    def exec_kernel(self, mode):
        cmd = ""
        
        if mode==0:
            cmd = "nvprof --csv --metrics gst_transactions ./{} 0 {}".format(self.k_fname, self.t_fname)
        elif mode==1:
            cmd = "./{} 1 {}".format(self.k_fname, self.t_fname)
        
        #print ("DEBUG cmd {}".format(cmd))

        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        out, err = p.communicate()
        if mode == 0:
            splitted_err = err.split(b',')
            idx = splitted_err.index(b'"Global Store Transactions"')
            data = np.empty( (3,), dtype=np.int64)
            for i in range(3):
                data[i] = splitted_err[idx+1+i]
                
            #print("{}".format(data))
            
            return data
        
        elif mode == 1:
            splitted_out = out[:-1].split(b"\n")
            data = np.empty( (len(splitted_out), 2), dtype=np.int64)
            for i in range(len(splitted_out)):
                data[i] = (np.fromstring(splitted_out[i], dtype=np.int64, sep=","))
            return data

        return None

    def compile(self):
        cmd = ("nvcc -Xptxas -dlcm=cg -o {} {}.cu".format(self.k_fname, self.k_fname))
        print ("DEBUG compiling...")
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        

    def step(self, act):

        assert act >= 0 and act <= 32
        """
        act = self.num2action(act)
        #print ("Action:[{}, {}]".format(act[0], act[1]))
        self.swap_tid(act[0], act[1])

        """
        done = False
        if act == 32:
            done = True
        else:

            t = np.reshape(self.t_map, (32, 32))
            row = t[act, act:].copy()
            col = t[act :, act].copy()
            t[act, act:] = col.copy()
            t[act:, act] = row.copy()
            self.t_map = np.reshape(t, (1024,))

        #print ("Outputting thread map...")
        np.savetxt(self.t_fname, self.t_map, fmt='%0.d')

        #print ("Executing kernel...")
        transactions = self.exec_kernel(0)
        obs = self.exec_kernel(1)
           
        #print ("Loading profiled data...")
        #data = self.load_profiled_data()

        #print ("Converting data into state...")
        state = self.addr2mem(obs)
        r = self.get_reward(transactions)
        
        return state, r, done, {} #Dummy

    def get_reward(self, transactions): 
        r = self.prev_transactions - transactions
        self.prev_transactions = transactions
        
        return r[2] # average

    def num2action(self, num):
        left = -1
        i = 1
        while num >= 0:
            num -= (self.t_num - i)
            left += 1
            i += 1
        right = self.t_num + num
        return [left, right]

#if __name__ == "__main__":
#    env = Environment(512, P_FNAME, T_FNAME, K_FNAME)
    #data = env.load_profiled_data()
    #print data
    #env.addr2mem(data)
#    state, r = env.step([0, 1])
#    print r
