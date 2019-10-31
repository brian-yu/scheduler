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

        prev_worker_host = None
        if self.job.prev_worker:
            prev_worker_host = self.job.prev_worker.host

        command = f"{Command.TRAIN.value} {job.job_name} {job.executable} {self.host} {prev_worker_host}"
        print(command)
        return self.send(command)

    def validate(self, job):
        if self.job:
            raise Exception("Currently assigned to a job.")
        self.job = job

        prev_worker_host = None
        if self.job.prev_worker:
            prev_worker_host = self.job.prev_worker.host

        command = f"{Command.VALIDATE.value} {job.job_name} {job.executable} {self.host} {prev_worker_host}"
        print(command)
        return self.send(command)

    def test(self, job):
        if self.job:
            raise Exception("Currently assigned to a job.")
        self.job = job

        prev_worker_host = None
        if self.job.prev_worker:
            prev_worker_host = self.job.prev_worker.host

        command = f"{Command.TEST.value} {job.job_name} {job.executable} {self.host} {prev_worker_host}"
        print(command)
        return self.send(command)

    def clean(self, job):
        command = f"{Command.CLEAN.value} {job.job_name}"
        print(command)
        return self.sendRecv(command)

    def __repr__(self):

        job = "Idle"
        if self.job:
            job = self.job.job_name
        
        return f"({self.host}: {job})"
