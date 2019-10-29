from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
import time
import argparse
import random
from collections import deque, defaultdict
from ftplib import FTP
from enum import Enum

from constants import Command, Status

class Mode(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"
    TESTING = "Testing"

# THESE MUST BE IPV4 ADDRESSES IN ORDER FOR FTP SERVER TO WORK.
PS_HOSTS = [
    '52.90.16.197',
    '3.91.26.174',
]

WORKER_HOSTS = [
    '54.172.145.68',
    '3.91.26.160',
    '18.234.228.46',
]

PS_MAX_JOBS = 5

class Client:
    def __init__(self, address_str):
        # Address of Daemon
        self.address = self.parse_address(address_str)
        self.host, self.port = self.address
        self.address_str = address_str

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

    def poll(self):
        status_str = self.sendRecv(Command.POLL.value)
        return Status(status_str)

    # TODO: Fix inheritance bug.
    def reset(self):
        self.job = None
        return self.sendRecv(Command.RESET.value)

class WorkerClient(Client):

    def __init__(self, address_str):
        Client.__init__(self, address_str)
        self.job = None

    def train(self, job):
        if self.job:
            raise Exception("Currently assigned to a job.")
        self.job = job

        ps_host = self.job.ps.tf_addr(self.job)
        prev_worker_host = None
        if self.job.prev_worker:
            prev_worker_host = self.job.prev_worker.tf_addr()

        command = f"{Command.TRAIN.value} {job.job_name} {ps_host} {self.tf_addr()} {prev_worker_host}"
        print(command)
        return self.send(command)

    def validate(self, job):
        if self.job:
            raise Exception("Currently assigned to a job.")
        self.job = job
        ps_host = self.job.ps.tf_addr(self.job)
        command = f"{Command.VALIDATE.value} {job.job_name} {ps_host} {self.tf_addr()}"
        print(command)
        return self.send(command)

    def test(self, job):
        if self.job:
            raise Exception("Currently assigned to a job.")
        self.job = job
        ps_host = self.job.ps.tf_addr(self.job)
        command = f"{Command.TEST.value} {job.job_name} {ps_host} {self.tf_addr()}"
        print(command)
        return self.send(command)

    def tf_addr(self):
        return f"{self.host}:2222"

    def __repr__(self):

        job = "Idle"
        if self.job:
            job = self.job.name
        
        return f"({self.host}: {job})"

class ParameterServerClient(Client):

    def __init__(self, address_str):
        Client.__init__(self, address_str)

        self.max_jobs = PS_MAX_JOBS
        
        # Keeps track of jobs using this node as a PS and their ports.
        self.job_ports = {}

    def start_ps(self, job, worker):
        self.allocate_job(job)
        command = f"{Command.START_PS.value} {job.job_name} {self.tf_addr(job)} {worker.tf_addr()}"
        print(command)
        return self.send(command)

    # TODO: Maybe consider if this needs to use sendRecv?
    def stop_ps(self, job):
        self.deallocate_job(job)
        return self.sendRecv(f"{Command.STOP_PS.value} {job.job_name}")

    def tf_addr(self, job):
        return f"{self.host}:{self.job_ports[job]}"

    def allocate_job(self, job):
        if len(self.job_ports) >= self.max_jobs:
            raise Exception("Too many jobs running on PS.")

        used_ports = set(self.job_ports.values())
        for port in range(2222, 3333):
            if port not in used_ports:
                self.job_ports[job] = port
                return

        raise Exception("Could not allocate port.")

    def deallocate_job(self, job):
        self.job_ports.pop(job)

    def can_allocate_job(self):
        return len(self.job_ports) <= self.max_jobs

    def __repr__(self):
        jobs = [f"<{port}: {job.name}>" for job, port in self.job_ports.items()]
        jobs_str = ", ".join(jobs)
        return f"({self.host}: [{jobs_str}])"


class Job:
    def __init__(self, job_name="default", epochs=3):
        self.job_name = job_name
        self.epochs = epochs
        self.curr_epoch = 0
        self.start_time = None
        self.completed = False

        self.worker = None
        self.prev_worker = None
        self.ps = None

    def assign_to(self, worker):
        self.prev_worker = self.worker
        self.worker = worker

    def increment_epoch(self):
        self.curr_epoch += 1
        if self.curr_epoch >= self.epochs:
            self.completed = True

    def __repr__(self):
        return f"({self.job_name}, curr_epoch={self.curr_epoch}, total_epochs={self.epochs})"


NUM_JOBS = 6
NUM_EPOCHS_LO = 2 # will be 25
NUM_EPOCHS_HI = 2 # will be 30

class Scheduler:

    def __init__(self, ps_hosts, worker_hosts, jobs=None):
        self.parameter_servers = [
            ParameterServerClient(f"{ps_host}:8888") for ps_host in ps_hosts]
        self.workers = [
            WorkerClient(f"{worker_host}:8888") for worker_host in worker_hosts]

        # Reset all workers and parameter servers.
        self.reset_nodes()

        self.jobs = jobs
        if not jobs:
        # Create NUM_JOBS jobs each with NUM_EPOCHS epochs
            get_num_epochs = lambda: random.randint(NUM_EPOCHS_LO, NUM_EPOCHS_HI)
            self.jobs = [
                Job(job_name=f"job_{i}", epochs=get_num_epochs()) for i in range(NUM_JOBS)]

        self.job_val_accs = {job: [] for job in self.jobs}

        # Assign parameter servers to jobs. Important! These should never change.
        for i, job in enumerate(self.jobs):
            job.ps = self.parameter_servers[i % len(self.parameter_servers)]

        # self.pending_jobs = deque(self.jobs)
        # self.currently_training_jobs = set()


    def run(self, mode=Mode.TRAINING):

        '''
        Invariants to keep in mind
            - need to limit number of jobs on a PS < 10 (8 to be safe)
            - job can only be running on 1 worker at a time
            - jobs need to be assigned to PS before being assigned to worker
        '''
        self.log(f"Beginning {mode.value}.")

        start_time = time.time()

        job_queue = deque(self.jobs)
        currently_running = set()

        
        log_interval = 60
        last_log_time = time.time() - log_interval

        while job_queue:

            if time.time() > last_log_time + log_interval:
                self.log(self.parameter_servers)
                self.log(self.workers)
                self.log(currently_running)

            for worker_id, worker in enumerate(self.workers):
                status = worker.poll()
                

                if status == Status.BUSY:
                    # Do something? Maybe do nothing?
                    pass

                elif status == Status.FREE:
                    # self.log(f"Worker_{worker_id} is {status}")
                    ## Cleanup current job on the worker if assigned.
                    if worker.job:
                        
                        prev_job = worker.job

                        self.log(f"Suspending {prev_job} on Worker_{worker_id}")
                        
                        # Stop PS for prev job
                        prev_job.ps.stop_ps(prev_job)

                        # Remove from set of currently training jobs
                        currently_running.remove(prev_job)

                        # Increment epoch
                        prev_job.increment_epoch()

                        self.log(f"Suspended job: {prev_job}")

                        if prev_job.completed:
                            job_queue.remove(prev_job)

                        worker.job = None


                    # Assign a job to this worker.
                    if job_queue:
                        job = job_queue.popleft()
                        job_queue.append(job)

                        if job not in currently_running and job.ps.can_allocate_job():
                            self.log(f"Assigning {job} to Worker_{worker_id}")
                            job.assign_to(worker)
                            currently_running.add(job)

                            job.ps.start_ps(job, worker)

                            if mode == Mode.TRAINING:
                                worker.train(job)
                            elif mode == Mode.VALIDATION:
                                worker.validate(job)
                            elif mode == Mode.TESTING:
                                worker.test(job)

            time.sleep(1)

        end_time = time.time()
        self.log(f"Finished {mode.value} in {end_time - start_time} seconds.")

    def train(self):
        self.run(mode=Mode.TRAINING)

    def validate(self):
        self.run(mode=Mode.VALIDATION)

    def test(self):
        self.run(mode=Mode.TESTING)

    # Reset all workers and parameter servers.
    def reset_nodes(self):
        for worker in self.workers:
            worker.reset()
        for ps in self.parameter_servers:
            ps.reset()

    def log(self, s):
        print(f"{time.ctime()}", end=" ")
        print(s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master.')
    args = parser.parse_args()

    scheduler = Scheduler(PS_HOSTS, WORKER_HOSTS)

    scheduler.train()
    scheduler.validate()
    scheduler.test()