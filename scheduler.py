import time
from datetime import datetime
import argparse
import random
from collections import deque
from ftplib import FTP
from enum import Enum
import os

from constants import Command, Status, Event
from clients import WorkerClient, ParameterServerClient

class Mode(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"
    TESTING = "Testing"

# THESE MUST BE IPV4 ADDRESSES IN ORDER FOR FTP SERVER TO WORK.
# PS_HOSTS = [
#     '52.90.16.197',
#     '3.91.26.174',
# ]

WORKER_HOSTS = [
    '54.172.145.68',
    '3.91.26.160',
    '18.234.228.46',
    '52.90.16.197',
    '3.91.26.174',
]

# PS_MAX_JOBS = 5

class Job:
    def __init__(self, job_name="default", epochs=3):
        self.job_name = job_name
        self.epochs = epochs
        self.curr_epoch = 0
        self.start_time = None
        self.completed = False

        self.worker = None
        self.prev_worker = None
        # self.ps = None

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


NUM_JOBS = 6
NUM_EPOCHS_LO = 25 # will be 25
NUM_EPOCHS_HI = 30 # will be 30

class Scheduler:

    def __init__(self, worker_hosts, jobs=None):
        # self.parameter_servers = [
        #     ParameterServerClient(f"{ps_host}:8888", max_jobs=PS_MAX_JOBS) for ps_host in ps_hosts]
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

        # self.job_val_accs = {job: [] for job in self.jobs}

        # Assign parameter servers to jobs. Important! These should never change.
        # for i, job in enumerate(self.jobs):
        #     job.ps = self.parameter_servers[i % len(self.parameter_servers)]

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
                    # f"{tab}Parameter servers:",
                    # f"{tab}\t{self.parameter_servers}",
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

                        # Delete unneeded checkpoint files from old worker.
                        prev_job.prev_worker.clean(prev_job)


                        # Log potential errors.
                        if time.time() - prev_job.start_time < 10:
                            warning = f"{job} ended {mode.value} less than 10 seconds after being started."
                            self.log(f"WARNING: {warning}")
                            self.warnings.append(warning)

                        
                        # Stop PS for prev job
                        # prev_job.ps.stop_ps(prev_job)

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

                            # job.ps.start_ps(job, worker)

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
        # for ps in self.parameter_servers:
        #     ps.reset()

    def log(self, s):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:", end=" ")
        print(s)

    # Download worker logs and test accuracy / loss
    def download_logs(self):
        for worker_id, worker in enumerate(self.workers):

            if os.environ['PUBLIC_IP'] and os.environ['PUBLIC_IP'] == worker.host:
                continue

            with FTP(worker.host, user="checkpoints", passwd="test") as ftp:
                # download worker log
                file_path = os.path.join('log_folder', 'worker_log')
                save_path = os.path.join('log_folder', f'worker_{worker_id}_log')
                self.create_dir(save_path)
                with open(save_path, 'wb') as fp:
                    ftp.retrbinary(f'RETR {file_path}', fp.write)

                # download accuracy files
                for acc_file in ftp.nlst('accuracy_folder'):
                    path = os.path.join('accuracy_folder', acc_file)
                    self.create_dir(path)
                    with open(path, 'wb') as fp:
                        ftp.retrbinary(f'RETR {path}', fp.write)

                # download loss files
                for acc_file in ftp.nlst('loss_folder'):
                    path = os.path.join('loss_folder', acc_file)
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

    scheduler = Scheduler(WORKER_HOSTS)

    scheduler.train()
    # scheduler.validate()
    scheduler.test()

    scheduler.download_logs()


    for warning in scheduler.warnings:
        print(warning)