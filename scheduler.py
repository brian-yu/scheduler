from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
import time
from datetime import datetime
import argparse
import random
from collections import deque, defaultdict
from ftplib import FTP
from enum import Enum
import os

from constants import Command, Status, Event

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
            job = self.job.job_name
        
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
        jobs = [f"<{port}: {job.job_name}>" for job, port in self.job_ports.items()]
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
        self.start_time = time.time()

    def increment_epoch(self):
        self.curr_epoch += 1
        if self.curr_epoch >= self.epochs:
            self.completed = True

    def __repr__(self):
        return f"({self.job_name}, epoch {self.curr_epoch + 1} of {self.epochs})"


NUM_JOBS = 3
NUM_EPOCHS_LO = 1 # will be 25
NUM_EPOCHS_HI = 1 # will be 30

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

        self.warnings = []


    def run(self, mode=Mode.TRAINING):

        '''
        Invariants to keep in mind
            - need to limit number of jobs on a PS < 10 (8 to be safe)
            - job can only be running on 1 worker at a time
            - jobs need to be assigned to PS before being assigned to worker
        '''
        self.log(f"===== Beginning {mode.value}.")

        start_time = time.time()

        job_queue = deque(self.jobs)
        currently_running = set()

        
        log_interval = 60
        last_log_time = time.time() - log_interval

        while job_queue:

            if time.time() > last_log_time + log_interval:
                tab = "\t\t\t"
                message = "\n".join([
                    "",
                    f"{tab}Parameter servers:",
                    f"{tab}\t{self.parameter_servers}",
                    f"{tab}Workers:",
                    f"{tab}\t{self.workers}",
                    f"{tab}Running jobs:",
                    f"{tab}\t{currently_running}",
                ])
                self.log(message)
                last_log_time = time.time()

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


                        # Log potential errors.
                        if time.time() - prev_job.start_time < 10:
                            warning = f"{job} ended {mode.value} less than 10 seconds after being started."
                            self.log(f"WARNING: {warning}")
                            self.warnings.append(warning)

                        
                        # Stop PS for prev job
                        prev_job.ps.stop_ps(prev_job)

                        # Remove from set of currently training jobs
                        currently_running.remove(prev_job)

                        # Increment epoch
                        if mode == Mode.TRAINING:
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
        self.log(f"===== Finished {mode.value} in {end_time - start_time} seconds.")

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
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:", end=" ")
        print(s)

    # Download worker logs and test accuracy / loss
    def download_logs(self):
        for worker_id, worker in enumerate(self.workers):
            with FTP(worker.host, user="checkpoints", passwd="test") as ftp:
                # download worker log
                file_path = os.path.join('log_folder', 'worker_log')
                save_path = os.path.join('log_folder', f'worker_{worker_id}_log')
                self.create_dir(save_path)
                with open(save_path, 'wb') as fp:
                    ftp.retrbinary(f'RETR {file_path}', fp.write)

                # download accuracy files
                for acc_file in ftp.nlst('/accuracy_folder'):
                    path = os.path.join('/accuracy_folder', acc_file)
                    self.create_dir(path)
                    with open(path, 'wb') as fp:
                        ftp.retrbinary(f'RETR {path}', fp.write)

                # download loss files
                for acc_file in ftp.nlst('/loss_folder'):
                    path = os.path.join('/loss_folder', acc_file)
                    self.create_dir(path)
                    with open(path, 'wb') as fp:
                        ftp.retrbinary(f'RETR {path}', fp.write)

    def create_dir(self, path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master.')
    args = parser.parse_args()

    scheduler = Scheduler(PS_HOSTS, WORKER_HOSTS)

    scheduler.train()
    scheduler.validate()
    scheduler.test()

    scheduler.download_logs()


    for warning in scheduler.warnings:
        print(warning)