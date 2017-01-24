import numpy as np
import subprocess

P_FNAME="mem.log"
T_FNAME="thread_map.csv"
K_FNAME="test"
WARP_SIZE=32
np.set_printoptions(threshold=np.inf)

# ====================
# Environment
# ====================
class Environment:
    def __init__(self, thread_num, p_fname, t_fname, k_fname):
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


        self.prev_transactions = 0
    def load_profiled_data(self):
        data = np.loadtxt(self.p_fname, delimiter=" ")
        return data

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

        print "DEBUG [%d, %d]"%(data.min(), data.max())
        # TODO segment size / data size
        width = 256 / 4 # scaled segment size w.r.t data size 
        height = data.max() - data.min() # address range
        depth = len(data) / WARP_SIZE # corresponding to warp axis

        print "DEBUG [%d,%d,%d]"%(width,height,depth)

        mem = np.zeros((depth, height, width), dtype=np.float16)

        # create binary image representing memory map
        for i in range(depth):
            offset = i*WARP_SIZE
            warp = data[offset:offset+WARP_SIZE]
            for j in range(WARP_SIZE):
                addr = warp[j]
                mem[i][addr/width][addr%width] = 1
        
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
            cmd = "nvprof --csv --metrics gld_transactions ./%s 0" % (self.k_fname)
        elif mode==1:
            cmd = "./%s 1" % (self.k_fname)
        
        print ("DEBUG cmd %s")%cmd

        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        out, err = p.communicate()
        if mode == 0:
            splitted_err = err.split(",")
            idx = splitted_err.index('"Global Load Transactions"')
            data = np.empty( (3,), dtype=np.int64)
            for i in range(3):
                data[i] = splitted_err[idx+1+i]
            return data
        elif mode == 1:
            splitted_out = out[:-1].split("\n")
            data = np.empty( (len(splitted_out), 2), dtype=np.int64)
            for i in range(len(splitted_out)):
                data[i] = (np.fromstring(splitted_out[i], dtype=np.int64, sep=","))
            return data

        
        return null

    def step(self, act):
        print ("Action:[%d, %d]")%(act[0], act[1])
        self.swap_tid(act[0], act[1])

        print ("Outputting thread map...")
        np.savetxt(self.t_fname, self.t_map, fmt='%0.d')

        print ("Executing kernel...")
        transactions = self.exec_kernel(0)
        obs = self.exec_kernel(1)
           
        #print ("Loading profiled data...")
        #data = self.load_profiled_data()

        print ("Converting data into state...")
        state = self.addr2mem(obs)
        r = self.get_reward(transactions)

        return state, r

    def get_reward(self, transactions): 
        r = self.prev_transactions - transactions
        self.prev_transactions = transactions
        
        return r[2] # average
                  

if __name__ == "__main__":
    env = Environment(512, P_FNAME, T_FNAME, K_FNAME)
    #data = env.load_profiled_data()
    #print data
    #env.addr2mem(data)
    state, r = env.step([0, 1])
    print r
