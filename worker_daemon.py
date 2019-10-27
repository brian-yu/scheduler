import argparse
import sys
import os
import time
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread, Lock
import subprocess
from enum import Enum

from daemon import Daemon

class Command(Enum):
    TRAIN = "TRAIN"
    START_PS = "START_PS"
    STOP_PS = "STOP_PS"


class WorkerDaemon(Daemon):

    def __init__(self, host, port, name="WORKER_DAEMON"):

        Daemon.__init__(self, host, port, name)

        self.job_ps_process = {}

    def receive(self, message):
        tokens = message.split()
        if not tokens:
            return "Invalid message."

        command_str = tokens[0]

        if Command(command_str) == Command.TRAIN:

            try:
                job, ps_hosts, worker_hosts = tokens[1:4]

                self.log(f"Training job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")
                # time.sleep(10)
                os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job}")
                self.log("Finished.")
            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.START_PS:
            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                proc = subprocess.Popen(
                    ['python3', 'task2.py',
                     f"--ps_hosts={ps_hosts}",
                     f"--worker_hosts={worker_hosts}",
                     "--job_name=ps --task_index=0",
                     f"--job={job}"],
                    shell=True)
                self.job_ps_process[job] = proc

            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.STOP_PS:
            try:
                job = tokens[1]
                job_proc = self.job_ps_process[job]
                job_proc.terminate()

            except Exception as err:
                self.log(f"Error: {err}")


        return "Done"



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Worker node daemon.')
    parser.add_argument("--port", default=8888, type=int, help="Port to listen for commands on.")
    args = parser.parse_args()

    worker_daemon = WorkerDaemon('', args.port)

    worker_daemon.listen()