import multiprocessing as mp
import time
import argparse
import sys
import os
import tensorflow as tf
import numpy as np

from worker import Worker
from ps import ParameterServer
from cluster import get_cluster


NUM_WORKERS = 2
NUM_PARAMETER_SERVERS = 2
# NUM_EPOCHS = 3

CLUSTER = get_cluster(NUM_PARAMETER_SERVERS, NUM_WORKERS)

print(CLUSTER)

def main():
    start = time.time()

    workers = []
    worker_queues = []
    master_queue = mp.Queue()

    # tasks = [i for i in range(100000)]

    # Create and start parameter servers
    # for ps_idx in range(NUM_PARAMETER_SERVERS):
    #     ps = ParameterServer(CLUSTER, ps_idx)
    #     ps_process = mp.Process(target=ps.run, args=())
    #     ps_process.daemon = True
    #     ps_process.start()

    for worker_idx in range(NUM_WORKERS):
        q = mp.Queue()
        worker_queues.append(q)
        worker = Worker(q, master_queue, CLUSTER, worker_idx)
        worker_process = mp.Process(target=worker.run, args=())
        workers.append(worker_process)
  
    for worker in workers:
        worker.daemon = True # Kill child process when master
        worker.start()

    # Communicate with workers here.
    # while tasks:
    #     for worker_queue in worker_queues:
    #         worker_queue.put(tasks.pop())

    # for worker_queue in worker_queues:
    #     worker_queue.put("END")

    
    # items = 0
    # while not master_queue.empty():
    #     master_queue.get()
    #     items += 1
    # print(f"Processed {items} items.")

    for worker in workers:
        worker.join()

    end = time.time()
  
    # both processes finished 
    print(f"Done in {end - start}s!") 

  
if __name__ == "__main__": 
    main()