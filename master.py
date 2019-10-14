from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
from time import sleep
import argparse
from collections import deque

from cluster import get_worker_addresses

class WorkerTracker:
    def __init__(self, address_str):
        self.address = self.parse_address(address_str)
        self.address_str = address_str
        self.task = None

    def parse_address(self, address):
        host, port = address.split(':')
        return (host, int(port))

    def sendRecv(self, message):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(self.address)
        sock.send(message.encode())
        reply = sock.recv(1024).decode()
        sock.close()
        # print(f"{message} -> {reply}")
        return reply

    def send(self, message):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(self.address)
        sock.send(message.encode())
        # print(f"{message}")
        sock.close()

    def reset(self):
        self.__init__(self.address_str)
        return self.send("RESET")

    def train(self, task):
        return self.send(f"TRAIN {task.job.job_name} {task.lo} {task.hi}")

    def status(self):
        return self.sendRecv("POLL")


class Task:
    def __init__(self, job, lo, hi):
        self.job = job
        self.lo = lo
        self.hi = hi
        # self.completed = False

class Job:
    def __init__(self, job_name="default", epochs=1, num_samples=4800, batch_size=120):
        self.job_name = job_name
        self.num_samples = num_samples
        self.batch_size = batch_size ### Multiple of 8

        # May want to shuffle this so that earlier samples are not favored.
        self.tasks = [Task(self, i, min(self.num_samples, i + self.batch_size) - 1) for i in range(0, self.num_samples, self.batch_size)] * epochs

        self.completed = False

class Master:

    def __init__(self, worker_addrs, jobs=[]):
        self.workers = [WorkerTracker(addr) for addr in worker_addrs]
        self.jobs = jobs

        job_tasks = [job.tasks for job in self.jobs]

        # ASSUMES THAT EACH JOB HAS THE SAME NUMBER OF TASKS
        self.task_queue = deque([task for tup in zip(*job_tasks) for task in tup])
        self.num_tasks = len(self.task_queue)
        self.completed_tasks = []

    def train(self):

        while len(self.completed_tasks) < self.num_tasks:
            completed = []
            for worker_id, worker in enumerate(self.workers):

                worker_status = worker.status()
                # print(f"Worker {worker_id} is {worker_status}")
                if worker_status == "BUSY":
                    continue

                # If worker is free, current task is completed.
                if worker.task:
                    self.completed_tasks.append(worker.task)
                    worker.task = None

                if self.task_queue:
                    task = self.task_queue.popleft()
                    worker.task = task
                    worker.train(task)
                    print(f"Running ({task.job.job_name}, [{task.lo}, {task.hi}]) on worker {worker_id}.")
            sleep(1)
        print("Finished training.")

    def parse_addr(self, addr):
        host, port = addr.split(':')
        return (host, int(port))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master.')
    parser.add_argument('num_workers', type=int, help='number of workers')
    args = parser.parse_args()
    print(args)

    print(f"{args.num_workers} workers.")

    worker_addrs = get_worker_addresses(args.num_workers)

    jobs = [Job(job_name=f"job_{i}") for i in range(3)]

    master = Master(worker_addrs, jobs=jobs)

    master.train()