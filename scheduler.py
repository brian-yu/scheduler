import time
from datetime import datetime
import argparse
import random
from collections import deque
from ftplib import FTP
from enum import Enum
import os

from constants import Command, Status, Event
from clients import WorkerClient
from file_utils import delete_directory_contents, create_directory

class Mode(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"
    TESTING = "Testing"

# THESE MUST BE IPV4 ADDRESSES IN ORDER FOR FTP SERVER TO WORK.

WORKER_HOSTS = [
    # '54.172.145.68',
    # '3.91.26.160',
    # '18.234.228.46',
    # '52.90.16.197',
    # '3.91.26.174',
    '54.86.29.194',
    '54.198.210.208',
]

class Job:
    def __init__(self, job_name="default", epochs=3, executable="alexnet.py"):
        self.job_name = job_name
        self.epochs = epochs
        self.curr_epoch = 0
        self.start_time = None
        self.completed = False
        self.executable = executable

        self.worker = None
        self.prev_worker = None

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

class Scheduler:

    def __init__(self, worker_hosts, jobs=None):
        self.workers = [
            WorkerClient(f"{worker_host}:8888") for worker_host in worker_hosts]

        # Reset all workers and parameter servers.
        self.reset_nodes()
        delete_directory_contents('log_folder')
        delete_directory_contents('accuracy_folder')
        delete_directory_contents('loss_folder')

        self.jobs = jobs
        if not jobs:
        # Create NUM_JOBS jobs each with NUM_EPOCHS epochs
            self.jobs = [
                Job(job_name=f"job_{i}", epochs=get_num_epochs()) for i in range(NUM_JOBS)]

        # self.job_val_accs = {job: [] for job in self.jobs}

        self.warnings = []


    def run(self, mode=Mode.TRAINING):

        '''
        Invariants to keep in mind
            - job can only be running on 1 worker at a time
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

                    ## Cleanup current job on the worker if assigned.
                    if worker.job:
                        
                        prev_job = worker.job

                        self.log(f"Suspending {prev_job} on Worker_{worker_id}")


                        # Log potential errors.
                        if time.time() - prev_job.start_time < 10:
                            warning = f"{job} ended {mode.value} less than 10 seconds after being started."
                            self.log(f"WARNING: {warning}")
                            self.warnings.append(warning)

                        

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

                        if job not in currently_running:
                            self.log(f"Assigning {job} to Worker_{worker_id}")
                            job.assign_to(worker)
                            currently_running.add(job)

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

    # Reset all workers.
    def reset_nodes(self):
        for worker in self.workers:
            worker.reset()

    def log(self, s):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:", end=" ")
        print(s)

    # Download worker logs and test accuracy / loss
    def download_logs(self):
        self.log("Downloading worker logs and test results.")

        for worker_id, worker in enumerate(self.workers):

            if 'PUBLIC_IP' in os.environ and os.environ['PUBLIC_IP'] == worker.host:
                continue

            with FTP(worker.host, user="checkpoints", passwd="test") as ftp:
                # download worker log
                file_path = os.path.join('log_folder', 'worker_log')
                save_path = os.path.join('log_folder', f'worker_{worker_id}_log')
                create_directory(save_path)
                with open(save_path, 'wb') as fp:
                    ftp.retrbinary(f'RETR {file_path}', fp.write)

                # download accuracy files
                for acc_file in ftp.nlst('accuracy_folder'):
                    path = os.path.join('accuracy_folder', acc_file)
                    create_directory(path)
                    with open(path, 'wb') as fp:
                        ftp.retrbinary(f'RETR {path}', fp.write)

                # download loss files
                for acc_file in ftp.nlst('loss_folder'):
                    path = os.path.join('loss_folder', acc_file)
                    create_directory(path)
                    with open(path, 'wb') as fp:
                        ftp.retrbinary(f'RETR {path}', fp.write)
        self.log("Finished downloading.")

    def save_logs(self):
        with open('log_folder/scheduler_log', 'w') as f:
            for job in self.jobs:
                data = [job.job_name, str(job.epochs), job.executable]
                f.write(" ".join(data) + "\n")

# NUM_CV_JOBS = 7

NUM_JOBS = 8
# NUM_EPOCHS_LO = 4 # will be 25
# NUM_EPOCHS_HI = 6 # will be 30


'''
Epoch recommandations:
- Alexnet: 25-30

- seq2seq: up to 100 -> `lstm_seq2seq.py`
- Transformer (transformer/main.py): 40 -> `transformer/main.py`
- LSTM text generator: 60 -> `lstm_text_generation.py`
'''

get_num_epochs = lambda lo, hi: random.randint(lo, hi)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master.')
    args = parser.parse_args()


    # job_exec = "lstm_seq2seq.py"
    # job_exec = "alexnet.py"
    jobs = []
    for i in range(NUM_JOBS // 4):
        job_id = len(jobs)
        jobs.append(
            Job(job_name=f"job_{job_id}",
                epochs=get_num_epochs(25, 30),
                executable='alexnet.py'))
    
    for i in range(NUM_JOBS // 4):
        job_id = len(jobs)
        jobs.append(
            Job(job_name=f"job_{job_id}",
                epochs=get_num_epochs(15, 20),
                executable='lstm_seq2seq.py'))

    for i in range(NUM_JOBS // 4):
        job_id = len(jobs)
        jobs.append(
            Job(job_name=f"job_{job_id}",
                epochs=get_num_epochs(15, 20),
                executable='transformer/main.py'))

    for i in range(NUM_JOBS // 4):
        job_id = len(jobs)
        jobs.append(
            Job(job_name=f"job_{job_id}",
                epochs=get_num_epochs(10, 15),
                executable='lstm_text_generation.py'))

    random.shuffle(jobs)

    scheduler = Scheduler(WORKER_HOSTS, jobs = jobs)

    scheduler.train()
    # scheduler.test()

    scheduler.download_logs()

    scheduler.save_logs()

    if scheduler.warnings:
        print("Warnings:")
        for warning in scheduler.warnings:
            print(f"\t{warning}")