import tensorflow as tf
import sys
import argparse

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


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('num_ps', type=int, help='number of parameter servers')
    parser.add_argument('num_workers', type=int, help='number of workers')
    parser.add_argument('ps_index', type=int, help='ps index')

    args = parser.parse_args()
    print(args)

    print(f"{args.num_ps} ps and {args.num_workers} workers.")

    cluster = get_cluster(args.num_ps, args.num_workers)
    print(cluster)

    server = tf.train.Server(cluster, job_name="ps", task_index=args.ps_index)

    server.join()

if __name__ == "__main__":
    main()