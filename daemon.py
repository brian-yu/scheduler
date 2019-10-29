import argparse
import sys
import os
import time
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread, Lock

class Daemon:

    def __init__(self, host, port, name="DAEMON"):

        self.host = host
        self.port = port
        self.name = name

        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)

        self.print_lock = Lock()

    def log(self, s):
        with self.print_lock:
            print(f"=== {time.ctime()} {self.name}:", end=" ")
            print(s)

    def receive(self, message):
        self.log(message)
        return "DONE"

    def handleClient(self, connection):
        while True:
            # try:
            data = connection.recv(1024)
            if data:
                # Set the response to echo back the recieved data 
                req = data.decode()
                # self.log(req)
                response = self.receive(req)
                connection.send(response.encode())
                connection.close()
                return True
            else:
                connection.close()
                raise Exception('Client disconnected')
            # except Exception as e:
            #     self.log(f"Unexpected error: {e}")
            #     connection.close()
            #     return False

    def same_node(self, host):
        ip = os.environ['PUBLIC_IP']
        return host == ip

    def listen(self, verbose=False):
        self.log(f"Listening on {self.host}:{self.port}")
        while True:
            connection, address = self.sock.accept()
            if verbose:
                self.log(f'Worker connected by {address} at {time.ctime()}')
            Thread(target = self.handleClient, args = (connection,)).start()
