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
import re
import glob
import errno

from daemon import Daemon
from constants import Command, Status, Event, Logger

class WorkerDaemon(Daemon):

    def __init__(self, host, port, name="WORKER_DAEMON"):

        Daemon.__init__(self, host, port, name)

        # self.job_ps_process = {}
        # self.job_worker_process = {}
        self.worker_status = Status.FREE
        self.worker_status_lock = Lock()

        self.logger = Logger()

    def receive(self, message):
        tokens = message.split()
        if not tokens:
            return "Invalid message."

        command_str = tokens[0]

        try:
            command = Command(command_str)
        except Exception as err:
            self.log(f"Error: {err}")
            return "Oops. Invalid command."

        if command == Command.TRAIN:

            try:
                job, executable, worker_host, prev_worker_host = tokens[1:]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY

                self.create_dir(f"checkpoints/{job}/")
                self.create_dir("log_folder/")
                self.create_dir("loss_folder/")
                self.create_dir("accuracy_folder/")

                self.log(f"Training job={job}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")
                '''
                For timing, maybe have task2.py log times in ./log_folder/times.txt.
                Then the worker daemon will read the times and send salient timing information.
                '''
                self.logger.log_event_start(job, Event.DOWNLOAD)

                # Transfer model files to current host.
                self.transfer_job_files(job, prev_worker_host)

                self.logger.log_event_end(job, Event.DOWNLOAD)

                os.system(f"python3 jobs/{executable} --job={job} --train")
                self.log("Training finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.VALIDATE:

            try:
                job, executable, worker_host, prev_worker_host = tokens[1:]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                # self.log(f"Validating job={job}, worker_host={worker_host}")
                self.log(f"Validating job={job}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")

                # Transfer model files to current host.
                self.transfer_job_files(job, prev_worker_host)

                os.system(f"python3 jobs/{executable} --job={job} --validate")
                self.log("Validation finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.TEST:

            try:
                job, executable, worker_host, prev_worker_host = tokens[1:]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                # self.log(f"Testing job={job}, worker_hosts={worker_hosts}")
                self.log(f"Testing job={job}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")

                # Transfer model files to current host.
                self.transfer_job_files(job, prev_worker_host)

                os.system(f"python3 jobs/{executable} --job={job} --test")
                self.log("Testing finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.POLL:
            try:
                with self.worker_status_lock:
                    status = self.worker_status
                return status.value

            except Exception as err:
                self.log(f"Error: {err}")

        elif command == Command.CLEAN:

            try:
                job = tokens[1]
                self.delete_directory_contents(f'checkpoints/{job}')

            except Exception as err:
                self.log(f"Error: {err}")


        elif command == Command.RESET:
            try:
                # self.terminate_parameter_servers()
                # Terminate workers
                os.system("kill -9 `ps -ef | grep task3.py | awk '{print $2}'`")
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

    # Download all files in checkpoint folder from prev_host.
    def transfer_job_files(self, job, prev_worker_host):
        if prev_worker_host == "None":
            return
        if self.same_node(prev_worker_host):
            return

        # Download files.
        self.download_checkpoint_files(job, prev_worker_host)

        # Delete from prev worker.
        self.sendRecv((prev_worker_host, 8888), f"{Command.CLEAN.value} {job}")

    def download_checkpoint_files(self, job, host):
        

        checkpoint_dir = os.path.join('checkpoints', job)
        self.create_dir(checkpoint_dir)

        ftp = FTP(host, user="checkpoints", passwd="test")
        ftp.cwd(checkpoint_dir)
        files = ftp.nlst()
        self.log(f"Downloading {files} from {host}.")
        for fname in files:
            path = os.path.join(checkpoint_dir, fname)
            with open(path, 'wb') as fp:
                ftp.retrbinary(f'RETR {fname}', fp.write)
        self.log(f"Finished downloading.")

    def cleanup(self, signal, frame):
        self.sock.close()
        self.log("Exiting.")
        sys.exit(0)

    def hostname(self, addr):
        self.log(f"Finding hostname for {addr}")
        return addr.split(":")[0]

    # if path = "folder/blah" or "folder/", will create folder.
    def create_dir(self, path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def delete_old_checkpoints(self, job):
        # Get all model.ckpt filenames
        ckpt_files = glob.glob(f'checkpoints/{job}/model.ckpt-*')
        if not ckpt_files:
            return
        self.log(f"ckpt files for {job}: {ckpt_files}")
        # Find latest version number
        ckpt_versions = sorted(
            [int(re.search('ckpt-(\d+).', file).group(1)) for file in ckpt_files])
        keep = ckpt_versions[-1]
        # Delete all ckpt files that are not the latest version
        for file in ckpt_files:
            if int(re.search('ckpt-(\d+).', file).group(1)) != keep:
                if os.path.isfile(file):
                    self.log(f"Deleting {file}")
                    os.remove(file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Worker node daemon.')
    parser.add_argument("--port", default=8888, type=int, help="Port to listen for commands on.")
    args = parser.parse_args()

    worker_daemon = WorkerDaemon('', args.port)

    signal.signal(signal.SIGINT, worker_daemon.cleanup)

    worker_daemon.listen()