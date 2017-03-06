from Environment import Environment
import numpy as np
from PIL import Image
P_FNAME="mem.log"
T_FNAME="thread_map_{}.csv"
K_FNAME="test"
THREAD_NUM=1024
env = Environment(THREAD_NUM, P_FNAME, T_FNAME.format(0), K_FNAME, comp=True)
state = env.reset()
#print ("{}".format(np.uint8(state[:,:,1])))
print ("{}".format(state.shape))

"""
img = Image.fromarray(np.uint8(state[:,:,13]), 'L')
img.show()
s_, r, done, _ = env.step(13)
#print("{}".format(r))
img2 = Image.fromarray(np.uint8(s_[:,:,13]), 'L')
img2.show()

#s_, r, done, _ = env.step(2)
img = Image.fromarray(np.uint8(state[:,:,10]), 'L')
img.show()
s_, r, done, _ = env.step(10)
#print("{}".format(r))
img2 = Image.fromarray(np.uint8(s_[:,:,10]), 'L')
img2.show()


img = Image.fromarray(np.uint8(state[:,:,15]), 'L')
img.show()
s_, r, done, _ = env.step(15)
#print("{}".format(r))
img2 = Image.fromarray(np.uint8(s_[:,:,15]), 'L')
img2.show()
"""
#img = Image.fromarray(np.uint8(state[:,:,16]), 'L')
#img.show()

for i in range(31):
    s_, r, done, _ = env.step(i)

#print ("{}".format(np.uint8(s_[:,:,0])))
    #print("{}".format(r))
#for i in range(31):
#    img2 = Image.fromarray(np.uint8(s_[:,:,i]), 'L')
#    img2.show()

    


