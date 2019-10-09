from socket import socket, AF_INET, SOCK_STREAM # portable socket api
import sys
import time
import os
from threading import Thread, Lock

class Worker:

    def __init__(self, host='', port=50000):
        self.num_samples = 1200
        self.curr_sample = 0
        self.completed = False
        self.status = "FREE"

        self.host = host
        self.port = port

        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)

        self.print_lock = Lock()
        self.status_lock = Lock()


    def train(self, batch_size):
        with self.status_lock:
            with self.print_lock:
                print(f"Status: {self.status}")
            if self.status == "BUSY":
                return self.completed
            self.status = "BUSY"

        begin = self.curr_sample
        end = min(self.num_samples, begin + batch_size)

        with self.print_lock:
            print(f"Begin training samples [{begin}, {end}).")

        for i in range(begin, min(self.num_samples, begin + batch_size)):
            self.curr_sample += 1

        time.sleep(5)

        if end == self.num_samples:
            self.completed = True
        
        with self.print_lock:
            print(f"Finish training samples [{begin}, {end}).")
            print(f"\t Completed? {self.completed}")

        with self.status_lock:
            self.status = "FREE"
        return self.completed

    def receive(self, message):

        tokens = message.split()
        command = tokens[0]
        with self.print_lock:
            print(tokens)

        if command == "RESET":
            self = self.__init__(self.host, self.port)
            return "True"
        elif command == "TRAIN":
            batch_size = int(tokens[1])
            self.train(batch_size)
            return str(self.completed)
        else: # POLL
            status = None
            with self.status_lock:
                status = self.status
            return f"{status} {self.completed}"

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