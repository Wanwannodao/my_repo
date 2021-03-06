import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../../"))

if import_path not in sys.path:
    sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker
from Environment import Environment

P_FNAME="mem.log"
T_FNAME="thread_map_{}.csv"
K_FNAME="test"

#VALID_ACTIONS = list(range(int(512*511/2))
VALID_ACTIONS = list(range(32 + 1))

#NUM_WORKERS = multiprocessing.cpu_count()-1
NUM_WORKERS = 2
MODEL_DIR = "/tmp/repo"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "simple/checkpoints")

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    print ("# of actions: {}".format(len(VALID_ACTIONS)))

    with tf.variable_scope("global") as vs:
        policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS))
        value_net = ValueEstimator(reuse=True)

    global_counter = itertools.count()

    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            name="worker_{}".format(worker_id),
            
            env=Environment(1024, P_FNAME, T_FNAME.format(worker_id), K_FNAME),
            
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            discount_factor=0.99,
            max_global_steps=None
        )
        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

    pe = PolicyMonitor(
        env=Environment(1024, P_FNAME, T_FNAME.format(NUM_WORKERS), K_FNAME),
        policy_net=policy_net,
        summary_writer=summary_writer,
        saver=saver)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, 8)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(900, sess, coord))
    monitor_thread.start()

    coord.join(worker_threads)
