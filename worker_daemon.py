import argparse
import sys
import os
import time
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread, Lock

from daemon import Daemon

class WorkerDaemon(Daemon):

    def __init__(self, host, port, name="WORKER_DAEMON"):

        Daemon.__init__(self, host, port, name)

    def receive(self, message):

        tokens = message.split()
        command = tokens[0]

        if command == "TRAIN":
            job, ps_hosts, worker_hosts = tokens[1:4]

            self.log(f"Training job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")
            # time.sleep(10)
            os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job}")
            self.log("Finished.")

            return ""

        return "Done"



if __name__ == "__main__":
    worker_daemon = WorkerDaemon('', 8888)

    worker_daemon.listen()