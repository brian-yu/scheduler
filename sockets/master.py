from socket import socket, AF_INET, SOCK_STREAM # portable socket api
from threading import Thread
import sys
from time import sleep

class WorkerTracker:
    def __init__(self, address_str):
        self.address = self.parse_address(address_str)
        print(self.address)
        self.address_str = address_str
        self.completed = False

    def parse_address(self, address):
        host, port = address.split(':')
        return (host, int(port))

    def sendRecv(self, message):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(self.address)
        sock.send(message.encode())
        reply = sock.recv(1024).decode()
        sock.close()
        print(f"{message} -> {reply}")
        return reply

    def send(self, message):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(self.address)
        sock.send(message.encode())
        print(f"{message}")
        sock.close()

    def reset(self):
        self.__init__(self.address_str)
        return self.send("RESET")

    def train(self, batch_size):
        return self.send(f"TRAIN {batch_size}")

    def poll(self):
        return self.sendRecv(f"POLL")

    def status(self):
        res = self.sendRecv("POLL")
        status, completed = res.split()
        return status, completed == "True"

class Master:

    def __init__(self, worker_addrs):
        self.batch_size = 9
        self.workers = [WorkerTracker(addr) for addr in worker_addrs]
        self.in_progress = {worker_id: worker for worker_id, worker in enumerate(self.workers)}
    
    def reset(self):
        for worker in self.workers:
            worker.reset()
        self = self.__init__([worker.address_str for worker in self.workers])

    def poll(self):
        for worker_id, worker in enumerate(self.workers):
            res = worker.poll()
            print(f"{worker_id}: {res}")

    def train(self):

        while self.in_progress:
            completed = []
            for worker_id, worker in self.in_progress.items():
                worker_status, worker_completed = worker.status()
                print(worker_id, worker_status, worker_completed)
                if worker_status == "BUSY":
                    continue
                if worker_completed:
                    completed.append(worker_id)
                    worker.completed = True
                    continue
                worker.train(self.batch_size)
                print(f"Training worker {worker_id}.")
                # if res == :
                #     worker.completed = True
                #     completed.append(worker_id)
                # else:
                #     worker.completed = False
            for worker_id in completed:
                self.in_progress.pop(worker_id)

            sleep(1)
        print("Finished training")

    def parse_addr(self, addr):
        host, port = addr.split(':')
        return (host, int(port))

if __name__ == "__main__":

    worker_addrs = sys.argv[1:]

    master = Master(worker_addrs)

    # master.reset()

    master.train()

    # master.poll()