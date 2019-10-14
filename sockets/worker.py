from socket import socket, AF_INET, SOCK_STREAM # portable socket api
import sys
import time
import os
from threading import Thread, Lock

class Worker:

    def __init__(self, host='', port=50000):
        self.status = "FREE"

        self.host = host
        self.port = port

        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)

        self.print_lock = Lock()
        self.status_lock = Lock()


    # TODO: train by job and allow async training of jobs.
    def train(self, job_name, lo, hi):
        with self.status_lock:
            with self.print_lock:
                print(f"Status: {self.status}")
            if self.status == "BUSY":
                return
            self.status = "BUSY"

        with self.print_lock:
            print(f"Began training job {job_name} on samples [{lo}, {hi}].")

        # Simulate training time
        time.sleep(3)
        
        with self.print_lock:
            print(f"Finished training job {job_name} on samples [{lo}, {hi}].")

        with self.status_lock:
            self.status = "FREE"

    def receive(self, message):

        tokens = message.split()
        command = tokens[0]

        if command == "RESET":
            self = self.__init__(self.host, self.port)
            return "True"
        elif command == "TRAIN":
            job_name = tokens[1]
            lo = int(tokens[2])
            hi = int(tokens[3])
            self.train(job_name, lo, hi)
            return str(self.status)
        else: # POLL
            status = None
            with self.status_lock:
                status = self.status
            return f"{status}"

    def handleClient(self, connection):
        while True:
            try:
                data = connection.recv(1024)
                if data:
                    # Set the response to echo back the recieved data 
                    req = data.decode()
                    with self.print_lock:
                        print(req)
                    response = self.receive(req)
                    connection.send(response.encode())
                    return True
                else:
                    raise Exception('Client disconnected')
            except Exception as e:
                with self.print_lock:
                    print("Unexpected error:", e)
                connection.close()
                return False

    def listen(self):
        while True:
            connection, address = self.sock.accept()
            with self.print_lock:
                print('Worker connected by', address, end=' ')
                print('at', time.ctime())
            Thread(target = self.handleClient, args = (connection,)).start()

if __name__ == "__main__":

    port = int(sys.argv[1])

    worker = Worker(port=port)
    worker.listen()