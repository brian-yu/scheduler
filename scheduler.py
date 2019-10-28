from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
import time
import argparse
from collections import deque

from constants import Command, Status


PS_HOSTS = [
    'ec2-52-90-16-197.compute-1.amazonaws.com',
]

WORKER_HOSTS = [
    'ec2-54-172-145-68.compute-1.amazonaws.com',
]



class WorkerClient:
    def __init__(self, address_str):
        self.address = self.parse_address(address_str)
        self.host, self.port = self.address
        self.address_str = address_str
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

    def train(self, job_name, ps_host, worker_host):
        self.job = job_name
        return self.send(f"{Command.TRAIN.value} {job_name} {ps_host} {worker_host}")

    def validate(self, job_name, ps_host, worker_host):
        self.job = job_name
        return self.send(f"{Command.VALIDATE.value} {job_name} {ps_host} {worker_host}")

    def start_ps(self, job_name, port, worker_host):
        return self.send(f"{Command.START_PS.value} {job_name} {self.host}:{port} {worker_host}")

    def stop_ps(self, job_name):
        return self.send(f"{Command.STOP_PS.value} {job_name}")

    def download_from_worker(self, job_name, prev_worker):
        pass

    def status(self):
        status = self.sendRecv(Command.POLL.value)
        return status

    def reset(self):
        self.job = None
        return self.sendRecv(Command.RESET.value)

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


NUM_JOBS = 1
NUM_EPOCHS = 2

class Scheduler:

    def __init__(self, ps_hosts, worker_hosts):
        self.parameter_servers = [WorkerClient(f"{ps_host}:8888") for ps_host in ps_hosts]
        self.workers = [WorkerClient(f"{worker_host}:8888") for worker_host in worker_hosts]

        for worker in self.workers:
            worker.reset()
        for ps in self.parameter_servers:
            ps.reset()

        self.jobs = [Job(job_name=f"job_{i}", epochs=NUM_EPOCHS) for i in range(NUM_JOBS)]

        for i, job in self.jobs:
            job.ps = self.parameter_servers[i % len(self.parameter_servers)]

        self.pending_jobs = deque(self.jobs)
        self.in_progress_jobs = set()


    def train(self):
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master.')
    args = parser.parse_args()

    ps_host = 'ec2-52-90-16-197.compute-1.amazonaws.com'
    worker_host = 'ec2-54-172-145-68.compute-1.amazonaws.com'

    ps0 = WorkerClient(f'{ps_host}:8888')
    worker0 = WorkerClient(f'{worker_host}:8888')


    print(ps0.reset())
    print(worker0.reset())


    print(ps0.status())
    print(worker0.status())

    ps0.start_ps('test', 2222, f'{worker_host}:2222')
    worker0.train('test', f'{ps_host}:2222', f'{worker_host}:2222')

    # worker0.validate('test', f'{ps_host}:2222', f'{worker_host}:2222')