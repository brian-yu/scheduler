from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
import time
import argparse
import random
from collections import deque, defaultdict

from constants import Command, Status

# THESE MUST BE IPV4 ADDRESSES IN ORDER FOR FTP SERVER TO WORK.
PS_HOSTS = [
    '52.90.16.197',
]

WORKER_HOSTS = [
    '54.172.145.68',
    '3.91.26.160',

]

PS_MAX_JOBS = 8

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


NUM_JOBS = 2
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

        # self.job_val_accs = defaultdict(list)

        # Assign parameter servers to jobs. Important! These should never change.
        for i, job in enumerate(self.jobs):
            job.ps = self.parameter_servers[i % len(self.parameter_servers)]

        self.pending_jobs = deque(self.jobs)
        self.currently_training_jobs = set()


    def train(self):

        '''
        Invariants to keep in mind
            - need to limit number of jobs on a PS < 10 (8 to be safe)
            - job can only be running on 1 worker at a time
            - jobs need to be assigned to PS before being assigned to worker
        '''

        start_time = time.time()

        while self.pending_jobs:
            for worker_id, worker in enumerate(self.workers):
                status = worker.poll()
                print(f"Worker_{worker_id} is {status}")

                if status == Status.BUSY:
                    # Do something? Maybe do nothing?
                    pass

                elif status == Status.FREE:
                    ## Cleanup current job on the worker if assigned.
                    if worker.job:
                        
                        prev_job = worker.job

                        print(f"Terminating {prev_job} on Worker_{worker_id}")
                        
                        # Stop PS for prev job
                        prev_job.ps.stop_ps(prev_job)
                        self.currently_training_jobs.remove(prev_job)

                        # Increment epoch
                        prev_job.increment_epoch()

                        print(f"Terminated job: {prev_job}")

                        if prev_job.completed:
                            self.pending_jobs.remove(prev_job)

                        worker.job = None


                    # Assign a job to this worker.
                    if self.pending_jobs:
                        job = self.pending_jobs.popleft()
                        self.pending_jobs.append(job)

                        if job not in self.currently_training_jobs:
                            print(f"Assigning {job} to Worker_{worker_id}")
                            job.assign_to(worker)
                            self.currently_training_jobs.add(job)
                            job.ps.start_ps(job, worker)
                            worker.train(job)

            time.sleep(1)

        end_time = time.time()
        print(f"Finished training in {end_time - start_time} seconds.")

    def validate(self):
        self.val_test()

    def test(self):
        self.val_test(mode="testing")

    def val_test(self, mode="validation"):
        job_q = deque(self.jobs)
        currently_testing = set()

        print(f"Beginning {mode}.")
        start_time = time.time()

        while job_q:
            for worker_id, worker in enumerate(self.workers):
                status = worker.poll()
                print(f"Worker_{worker_id} is {status}")

                if status == Status.BUSY:
                    # Do something? Maybe do nothing?
                    pass

                elif status == Status.FREE:
                    ## Cleanup current job on the worker if assigned.
                    if worker.job:
                        
                        prev_job = worker.job

                        print(f"Terminating {prev_job} on Worker_{worker_id}")
                        
                        # Stop PS for prev job
                        prev_job.ps.stop_ps(prev_job)
                        currently_testing.remove(prev_job)

                        print(f"Terminated job: {prev_job}")

                        if prev_job.completed:
                            job_q.remove(prev_job)

                        worker.job = None


                    # Assign a job to this worker.
                    if job_q:
                        job = job_q.popleft()
                        job_q.append(job)

                        if job not in currently_testing:
                            print(f"Begin {mode} of {job} on Worker_{worker_id}")
                            job.assign_to(worker)
                            currently_testing.add(job)
                            job.ps.start_ps(job, worker)
                            if mode == "testing":
                                worker.test(job)
                            else:
                                worker.validate(job)

            time.sleep(1)
        end_time = time.time()
        print(f"Finished {mode} in {end_time - start_time} seconds.")


    # Reset all workers and parameter servers.
    def reset_nodes(self):
        for worker in self.workers:
            worker.reset()
        for ps in self.parameter_servers:
            ps.reset()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master.')
    args = parser.parse_args()

    # ps_host = '52.90.16.197'
    # worker_host = '54.172.145.68'

    # ps0 = ParameterServerClient(f'{ps_host}:8888')
    # worker0 = WorkerClient(f'{worker_host}:8888')


    # print(ps0.reset())
    # print(worker0.reset())


    # print(ps0.poll())
    # print(worker0.poll())

    # # ps0.start_ps('test', 2222, f'{worker_host}:2222')
    # # worker0.train('test', f'{ps_host}:2222', f'{worker_host}:2222')

    # worker0.validate('test', f'{ps_host}:2222', f'{worker_host}:2222')

    scheduler = Scheduler(PS_HOSTS, WORKER_HOSTS)

    scheduler.train()
    scheduler.validate()
    # scheduler.test()