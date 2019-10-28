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

        try:
            command = Command(command_str)
        except Exception as err:
            self.log(f"Error: {err}")
            return "Oops. Invalid command."

        if command == Command.TRAIN:

            try:
                job, ps_host, worker_host, prev_worker_host = tokens[1:5]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Training job={job}, ps_hosts={ps_host}, worker_hosts={worker_host}, prev_worker_host={prev_worker_host}")
                '''
                For timing, maybe have task2.py log times in ./log_folder/times.txt.
                Then the worker daemon will read the times and send salient timing information.
                '''
                self.download_train_files(job, ps_host, prev_worker_host)
                os.system(f"python3 task2.py --ps_hosts={ps_host} --worker_hosts={worker_host} --job_name=worker --task_index=0 --job={job} --train")
                self.log("Training finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.VALIDATE:

            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Validating job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")

                # Download 'latest_model_{jobName}.ckpt' .index and .data files.
                self.download_latest_model(job, ps_hosts)

                os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job} --validate")
                self.log("Validation finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.TEST:

            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                with self.worker_status_lock:
                    self.worker_status = Status.BUSY
                self.log(f"Testing job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")

                # Download 'latest_model_{jobName}.ckpt' .index and .data files.
                self.download_latest_model(job, ps_hosts)

                os.system(f"python3 task2.py --ps_hosts={ps_hosts} --worker_hosts={worker_hosts} --job_name=worker --task_index=0 --job={job} --test")
                self.log("Testing finished.")
            except Exception as err:
                self.log(f"Error: {err}")
            finally:
                with self.worker_status_lock:
                    self.worker_status = Status.FREE

        elif command == Command.START_PS:
            try:
                job, ps_hosts, worker_hosts = tokens[1:4]
                self.log(f"Starting PS for job={job}, ps_hosts={ps_hosts}, worker_hosts={worker_hosts}")
                proc = subprocess.Popen(
                    ['python3', 'task2.py', f"--ps_hosts={ps_hosts}", f"--worker_hosts={worker_hosts}",
                     "--job_name=ps", "--task_index=0", f"--job={job}"])
                self.job_ps_process[job] = proc

            except Exception as err:
                self.log(f"Error: {err}")

        elif command == Command.STOP_PS:
            try:
                job = tokens[1]

                self.log(f"Killing PS for job={job}")
                job_proc = self.job_ps_process[job]
                job_proc.terminate()
                self.job_ps_process.pop(job)

            except Exception as err:
                self.log(f"Error: {err}")

        elif command == Command.POLL:
            try:
                with self.worker_status_lock:
                    status = self.worker_status
                return status.value

            except Exception as err:
                self.log(f"Error: {err}")

        elif command == Command.RESET:
            try:
                self.terminate_parameter_servers()
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
    def download_train_files(self, job, ps_host, prev_worker_host):

        # If this is the first worker running a job, we don't need to download
        # any files.
        if prev_worker_host == "None":
            return

        ps = self.hostname(ps_host)
        prev_worker = self.hostname(prev_worker_host)

        # Need to Download checkpoint and .meta files first, so that we can read them 
        # to determine which .index file to download from the PS.
        if prev_worker != self.host:
            fnames = ['checkpoint']
            self.download_files(job, prev_worker, fnames)

        # Download .index file from PS
        with open(f"checkpoints/{job}/checkpoint") as f:
            full_path = f.readline().rstrip("\n").rstrip("\"").split(":")[1]
            self.log(f"CKPT_PATH={full_path}")
            ckpt = full_path.split("/")[-1]
            self.log(f"CKPT_FILE={ckpt}")
            index = f"{ckpt}.index"
            meta = f"{ckpt}.meta"
            # Download .index file from ps
            self.download_files(job, ps, [index])
            # Download .meta file from prev_worker
            if prev_worker != self.host:
                self.download_files(job, prev_worker, [meta])

    def download_latest_model(self, job, ps_hosts):
        # Download 'latest_model_{jobName}.ckpt' .index and .data files.
        fnames = [f"latest_model_{job}.ckpt.index", f"latest_model_{job}.ckpt.data-00000-of-00001"]
        ps_host = ps_hosts.split(":")[0]
        self.download_files(job, ps_host, fnames)

    def download_files(self, job, host, fnames):
        self.log(f"Downloading {fnames} from {host}")
        ftp = FTP(host, user="checkpoints", passwd="test")
        ftp.cwd(job)
        with self.print_lock:
            ftp.dir()
        for fname in fnames:
            path = f'checkpoints/{job}/{fname}'
            self.create_dir(path)
            with open(path, 'wb') as fp:
                ftp.retrbinary(f'RETR {fname}', fp.write)
        self.log(f"Downloaded {fnames} from {host}.")


    def terminate_parameter_servers(self):
        self.log(f"Killing {len(self.job_ps_process)} PS processes.")
        for job, ps_proc in self.job_ps_process:
            self.log(f"Killing PS for {job}")
            ps_proc.terminate()
        self.job_ps_process = {}

    def cleanup(self, signal, frame):
        self.terminate_parameter_servers()
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Worker node daemon.')
    parser.add_argument("--port", default=8888, type=int, help="Port to listen for commands on.")
    args = parser.parse_args()

    worker_daemon = WorkerDaemon('', args.port)

    signal.signal(signal.SIGINT, worker_daemon.cleanup)

    worker_daemon.listen()