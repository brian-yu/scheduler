from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from ftplib import FTP
from enum import Enum
import os

from constants import Command, Status

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

    def __init__(self, address_str, max_jobs=5):
        Client.__init__(self, address_str)

        self.max_jobs = max_jobs
        
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