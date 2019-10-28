import argparse
import sys
import os
import shutil
import time
import signal
import subprocess
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread, Lock
from ftplib import FTP

from daemon import Daemon
from constants import Command, Status

class WorkerDaemon(Daemon):

    def __init__(self, host, port, name="WORKER_DAEMON"):

        Daemon.__init__(self, host, port, name)

        self.job_ps_process = {}
        # self.job_worker_process = {}
        self.worker_status = Status.FREE
        self.worker_status_lock = Lock()

    def receive(self, message):
        tokens = message.split()
        if not tokens:
            return "Invalid message."

        command_str = tokens[0]

        if Command(command_str) == Command.TRAIN:

            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Training job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")
                '''
                For timing, maybe have task2.py log times in ./log_folder/times.txt.
                Then the worker daemon will read the times and send salient timing information.
                '''
                os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job} --train")
                with self.worker_status_lock:
                    self.worker_status = Status.FREE
                self.log("Finished.")
            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.VALIDATE:

            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Validating job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")

                # Download 'latest_model_{jobName}.ckpt' .index and .data files.
                self.download_latest_model(job, ps_hosts)

                os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job} --validate")
                with self.worker_status_lock:
                    self.worker_status = Status.FREE
                self.log("Finished.")
            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.TEST:

            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Testing job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")

                # Download 'latest_model_{jobName}.ckpt' .index and .data files.
                self.download_latest_model(job, ps_hosts)

                os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job} --test")
                with self.worker_status_lock:
                    self.worker_status = Status.FREE
                self.log("Finished.")
            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.START_PS:
            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                self.log(f"Starting PS for job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")
                proc = subprocess.Popen(
                    ['python3', 'task2.py', f"--ps_hosts={ps_hosts}", f"--worker_hosts={worker_hosts}",
                     "--job_name=ps", "--task_index=0", f"--job={job}"])
                self.job_ps_process[job] = proc

            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.STOP_PS:
            try:
                job = tokens[1]

                self.log(f"Killing PS for job={job}")
                job_proc = self.job_ps_process[job]
                job_proc.terminate()

            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.POLL:
            try:
                with self.worker_status_lock:
                    status = self.worker_status
                return status.value

            except Exception as err:
                self.log(f"Error: {err}")

        elif Command(command_str) == Command.RESET:
            try:
                self.delete_directory_contents('checkpoints')
                self.delete_directory_contents('log_folder')
                self.delete_directory_contents('loss_folder')
                self.delete_directory_contents('accuracy_folder')
            except Exception as err:
                self.log(f"Error: {err}")
            
        return "Done"

    def delete_directory_contents(self, rel_path):
        path = os.path.join(os.getcwd(), rel_path)
        self.log(f"Deleting {path}.")
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def download_latest_model(self, job, ps_hosts):
        # Download 'latest_model_{jobName}.ckpt' .index and .data files.
        fnames = [f"latest_model_{job}.ckpt.index", f"latest_model_{job}.ckpt.data-00000-of-00001"]
        ps_host = ps_hosts.split(":")[0]
        self.log(f"Downloading {fnames} from {ps_host}")
        ftp = FTP(ps_host, user="checkpoints", passwd="test")
        ftp.cwd(job)
        for fname in fnames:
            with open(f'checkpoints/test/{fname}', 'wb') as fp:
                ftp.retrbinary(f'RETR {fname}', fp.write)

    def cleanup(self, signal, frame):
        self.log(f"Killing {len(self.job_ps_process)} PS processes.")
        for proc in self.job_ps_process.values():
            proc.terminate()
        self.sock.close()
        self.log("Exiting.")
        sys.exit(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Worker node daemon.')
    parser.add_argument("--port", default=8888, type=int, help="Port to listen for commands on.")
    args = parser.parse_args()

    worker_daemon = WorkerDaemon('', args.port)

    signal.signal(signal.SIGINT, worker_daemon.cleanup)

    worker_daemon.listen()