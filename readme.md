# Multithreaded TensorFlow scheduler

## How to run
1. Start the parameter server. `python ps.py 1 2 0` starts 1 parameter server configured to handle 2 workers with the current parameter server being the 0th index.

2. Start the workers. If you want to start 2 workers that share 1 parameter server, run `python worker.py --num_ps 1 --num_workers 2 --worker_index 0 --listen` and `python worker.py --num_ps 1 --num_workers 2 --worker_index 1 --listen` in separate tabs.

3. Run the master that will coordinate training. `python master.py 2` runs the master configured to communicate with 2 workers.

