#!/usr/bin/env python3

import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

recv = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'Hello, world')
    recv.append(repr(s.recv(1024)))
    s.sendall(b'Hello, world')
    recv.append(repr(s.recv(1024)))
    s.sendall(b'TRAIN 100 200')
    recv.append(repr(s.recv(1024)))


print('Received', recv)