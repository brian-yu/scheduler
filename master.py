from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
import time
import argparse
from collections import deque

from cluster import get_worker_addresses

class WorkerTracker:
    def __init__(self, address_str):
        self.address = self.parse_address(address_str)
        self.address_str = address_str
        self.task = None
        self.job = None

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

    def train(self, job):
        self.job = job
        self.job.start_time = time.time()
        return self.send(f"TRAIN {job.job_name} {job.curr_sample}")

    def train_interrupt(self):
        return self.sendRecv(f"TRAIN_INTERRUPT")

    def status(self):
        status, job, sample_idx = self.sendRecv("POLL").split()
        if sample_idx != 'None':
            sample_idx = int(sample_idx)
        return status, job, sample_idx

class Job:
    def __init__(self, job_name="default", epochs=1, num_samples=4800, batch_size=120):
        self.job_name = job_name
        self.num_samples = num_samples
        self.curr_sample = 0
        self.start_time = None
        self.completed = False

    def set_curr_sample(self, i):
        self.curr_sample = i
        if self.curr_sample == self.num_samples - 1:
            self.completed = True

class Master:

    def __init__(self, worker_addrs, jobs=[]):
        self.workers = [WorkerTracker(addr) for addr in worker_addrs]
        self.jobs = jobs
        self.jobs_by_name = {job.job_name: job for job in self.jobs}
        self.num_jobs = len(self.jobs)

        self.pending_jobs = deque(self.jobs)
        self.completed_jobs = set()

        self.train_interval = 60 # how long to train each job for in seconds before suspending

        # job_tasks = [job.tasks for job in self.jobs]

        # ASSUMES THAT EACH JOB HAS THE SAME NUMBER OF TASKS
        # self.task_queue = deque([task for tup in zip(*job_tasks) for task in tup])
        # self.num_tasks = len(self.task_queue)
        # self.completed_tasks = []

    def train(self):

        while self.pending_jobs:

            # completed = []

            for worker_id, worker in enumerate(self.workers):

                status, last_job_name, last_sample = worker.status()
                # print(status, last_job_name, last_sample)
                if status == "BUSY":
                    # TODO: SUSPEND JOB IF DESIRED HERE
                    if time.time() - worker.job.start_time >= self.train_interval:
                        print(f"Suspending {worker.job.job_name} on worker {worker_id}.")
                        worker.train_interrupt()
                    continue
                elif status == "STOPPING":
                    continue
                elif status == "FREE":
                    if last_job_name != 'None':
                        last_job = self.jobs_by_name[last_job_name]
                        last_job.set_curr_sample(last_sample + 1)
                        print(f"Updating status of {last_job.job_name} to sample {last_sample+1}.")
                        if last_job.completed:
                            self.pending_jobs.remove(last_job)
                            self.completed_jobs.add(last_job)

                    if self.pending_jobs:
                        job = self.pending_jobs.popleft()
                        self.pending_jobs.append(job)
                        # task = self.task_queue.popleft()
                        # worker.task = task
                        # worker.train(task)
                        worker.train(job)
                        print(f"Running ({job.job_name}, {job.curr_sample}) on worker {worker_id}.")

            time.sleep(1)
        print("Finished training.")

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