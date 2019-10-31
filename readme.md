# Distributed deep learning job scheduler

Important files
- `worker-daemon.py`: Runs a socket server that listens for commands and runs training tasks in a thread.
- `scheduler.py`: Initializes jobs and tasks and decides how to schedule tasks on workers.
- `jobs/`: Deep learning job code.
- `data/`: Data for training tasks.

## How to run

0. On each worker, install required dependencies using `./install.sh`.

1. On each worker, start the FTP server and daemon using `./run.sh`.

2. Run `python3 scheduler.py` to begin training.