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
                job, worker_host, prev_worker_host = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Training job={job}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")
                '''
                For timing, maybe have task2.py log times in ./log_folder/times.txt.
                Then the worker daemon will read the times and send salient timing information.
                '''
                self.logger.log_event_start(job, Event.DOWNLOAD)
                self.download_latest_model(job, prev_worker_host)
                # self.sendRecv((prev_worker_host, 8888), f"{Command.CLEAN.value} {job}")

                self.logger.log_event_end(job, Event.DOWNLOAD)

                os.system(f"python3 task3.py --job={job} --train")
                self.log("Training finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.VALIDATE:

            try:
                job, worker_host, prev_worker_host = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                # self.log(f"Validating job={job}, worker_host={worker_host}")
                self.log(f"Validating job={job}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")

                # Download 'latest_model_{jobName}.ckpt' .index and .data files.
                self.download_latest_model(job, prev_worker_host)
                # self.sendRecv((prev_worker_host, 8888), f"{Command.CLEAN.value} {job}")

                os.system(f"python3 task3.py --job={job} --validate")
                self.log("Validation finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.TEST:

            try:
                job, worker_host, prev_worker_host = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                # self.log(f"Testing job={job}, worker_hosts={worker_hosts}")
                self.log(f"Testing job={job}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")

                # Download 'latest_model_{jobName}.ckpt' .index and .data files.
                self.download_latest_model(job, prev_worker_host)
                # self.sendRecv((prev_worker_host, 8888), f"{Command.CLEAN.value} {job}")

                os.system(f"python3 task3.py --job={job} --test")
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

    # TODO: Download `checkpoint` and `.meta` files from prev worker.
    # Reads checkpoint file and downloads appropriate .index file from PS.
    # TODO: don't download checkpoint and .meta if prev_worker is current worker?
    def download_train_files(self, job, prev_worker_host):

        # If this is the first worker running a job, we don't need to download
        # any files.
        if prev_worker_host == "None":
            return

        ps = self.hostname(ps_host)
        prev_worker = self.hostname(prev_worker_host)

        # Need to download checkpoint file first, so that we can read it 
        # and determine which .index file to download from the PS and which .meta
        # file to download.
        self.log(f"Same as previous worker? {self.same_node(prev_worker)}")
        if not self.same_node(prev_worker):
            fnames = ['checkpoint']
            self.download_checkpoint_files(job, prev_worker, fnames)

        # Download .index file from PS and .meta file from prev worker.
        with open(f"checkpoints/{job}/checkpoint") as f:
            full_path = f.readline().rstrip("\n").rstrip("\"").split(":")[1]
            self.log(f"CKPT_PATH={full_path}")
            ckpt = full_path.split("/")[-1]
            self.log(f"CKPT_FILE={ckpt}")
            index = f"{ckpt}.index"
            meta = f"{ckpt}.meta"
            # Download .index file from ps
            self.download_checkpoint_files(job, ps, [index])
            # Download .meta file from prev_worker
            if not self.same_node(prev_worker):
                self.download_checkpoint_files(job, prev_worker, [meta])

    # def download_latest_model(self, job, ps_hosts):
    #     # Download 'latest_model_{jobName}.ckpt' .index and .data files.
    #     fnames = [f"latest_model_{job}.ckpt.index", f"latest_model_{job}.ckpt.data-00000-of-00001"]
    #     ps_host = ps_hosts.split(":")[0]
    #     self.download_checkpoint_files(job, ps_host, fnames)

    def download_latest_model(self, job, prev_worker_host):
        if prev_worker_host == "None":
            return
        if self.same_node(prev_worker_host):
            return
        # Download 'latest_model_{jobName}.ckpt' .index and .data files.
        fnames = [
            f"latest_model_{job}.ckpt.index",
            f"latest_model_{job}.ckpt.meta",
            f"latest_model_{job}.ckpt.data-00000-of-00001"]

        # Download files.
        self.download_checkpoint_files(job, prev_worker_host, fnames)

        # Delete from prev worker.
        self.sendRecv((prev_worker_host, 8888), f"{Command.CLEAN.value} {job}")

    def download_checkpoint_files(self, job, host, fnames):
        self.log(f"Downloading {fnames} from {host}")
        ftp = FTP(host, user="checkpoints", passwd="test")
        ftp.cwd(os.path.join('checkpoints', job))
        with self.print_lock:
            ftp.dir()
        for fname in fnames:
            path = f'checkpoints/{job}/{fname}'
            self.create_dir(path)
            with open(path, 'wb') as fp:
                ftp.retrbinary(f'RETR {fname}', fp.write)
        self.log(f"Downloaded {fnames} from {host}.")

    def cleanup(self, signal, frame):
        self.sock.close()
        self.log("Exiting.")
        sys.exit(0)

    def hostname(self, addr):
        self.log(f"Finding hostname for {addr}")
        return addr.split(":")[0]

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