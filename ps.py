import tensorflow as tf
import sys

from cluster import get_cluster

class ParameterServer:

    def __init__(self, cluster, ps_idx):
        self.cluster = cluster
        self.idx = ps_idx
        self.server = tf.train.Server(cluster,
                        job_name="ps",
                        task_index=self.idx)


    def run(self):
        self.log("Running.")
        self.server.join()

    def log(self, s):
        print(f"===== PS_{self.idx}:", end=" ")
        print(s)

def main():

    if len(sys.argv) < 4:
        print("Please specify num_ps, num_workers, and ps_idx.")
        return

    num_ps = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    ps_idx = int(sys.argv[3])

    print(f"{num_ps} ps and {num_workers} workers.")

    cluster = get_cluster(num_ps, num_workers)
    print(cluster)

    server = tf.train.Server(cluster, job_name="ps", task_index=ps_idx)

    server.join()

if __name__ == "__main__":
    main()