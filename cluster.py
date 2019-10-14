import tensorflow as tf



def get_cluster(num_ps, num_workers):
    return tf.train.ClusterSpec(
        {
            "ps": [f"localhost:{port}" for port in range(2222, 2222 + num_ps)],
            "worker": [f"localhost:{port}" for port in range(
                2222 + num_ps, 2222 + num_ps + num_workers)]
        })



def get_worker_port(worker_index):
    return 2323 + worker_index

def get_worker_addresses(num_workers):
    return [f"localhost:{2323 + i}" for i in range(num_workers)]