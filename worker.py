from sklearn.metrics import roc_auc_score
import cv2
import argparse
import sys
import os
import zipfile
import tensorflow as tf
import numpy as np
import time
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread, Lock

from cluster import get_cluster, get_worker_port

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session


IMG_SIZE_ALEXNET = 227
validating_size = 40
nodes_fc1 = 4096
nodes_fc2 = 4096
output_classes = 4

TRAIN_DIR = os.getcwd()

class Worker:
    # Initialize the Worker. Keep this in sync with reset().
    def __init__(self, host, port, cluster, worker_idx, job_name="default"):
        self.host = host
        self.port = port
        self.cluster = cluster
        self.idx = worker_idx
        self.job_name = job_name

        self.status = "FREE"
        self.train_interrupt = False
        self.last_trained_job = None
        self.last_trained_sample = None

        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)

        self.print_lock = Lock()
        self.status_lock = Lock()
        self.train_interrupt_lock = Lock()
        self.last_trained_lock = Lock()

        self.step_size = 8
        self.log(cluster)
        self.log(self.idx)
        self.server = tf.distribute.Server(cluster,
                         job_name="worker",
                         task_index=self.idx)

        self.cumulative_time_saved = 0

        # Load training and test data.
        self.load_data()
        # Define model layers.
        self.build_model(verbose=True)

        # The StopAtStepHook handles stopping after running given steps.
        self.hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        #init = tf.global_variables_initializer()
        self.acc_list = []
        self.auc_list = []
        self.loss_list = []
        # self.saver = tf.train.Saver()

        #GPU settings
        self.session_conf = tf.ConfigProto()
        # self.session_conf = tf.ConfigProto(allow_soft_placement=True)
        # self.session_conf.gpu_options.allow_growth = True
        # self.session_conf.gpu_options.allocator_type = 'BFC'

        self.log(f"Last sample: {self.steps-self.remaining - 1}, Step size: {self.step_size}")

    def train(self, job_name="default", lo = 0, hi=None):
        hi = self.steps - self.remaining
        with self.status_lock:
            self.log(f"Status: {self.status}")
            if self.status == "BUSY" or lo >= hi:
                return
            self.status = "BUSY"

        self.set_job(job_name)
        self.log(f"Starting training job {self.job_name} beginning at sample {lo}.")
        self.log(f"Train folder: {self.train_folder}")
        self.log(f"Job name: {self.job_name}")

        if not lo:
            lo = 0

        build_start = time.time()
        self.build_model()
        build_end = time.time()
        self.cumulative_time_saved += build_end - build_start

        self.log_time(self.job_name, 'start', time.time() - self.cumulative_time_saved)

        saver = tf.train.Saver()

        assert self.job_name in self.train_folder, "Incorrect train folder"

        with tf.train.MonitoredTrainingSession(
                master=self.server.target,
                is_chief=(self.idx == 0),
                checkpoint_dir=self.train_folder,
                hooks=self.hooks,
                config=self.session_conf) as mon_sess:
            self.log("Starting training loop.")
            for j in range(lo, hi, self.step_size):
                #Feeding step_size-amount data with 0.5 keeping probabilities on DROPOUT LAYERS
                # print(j)
                _, c = mon_sess.run(
                    [self.train_op, self.cross_entropy],
                    feed_dict={
                        self.x: self.X[j:j + self.step_size],
                        self.y_true: self.Y[j:j + self.step_size],
                        self.hold_prob1: 0.5,
                        self.hold_prob2: 0.5
                    })

                with self.train_interrupt_lock:
                    if self.train_interrupt:
                        self.log("Breaking training loop.")
                        break
                if j % 200 == 0:
                    self.log(f"Trained sample {j}.")

                # if (self.global_step.eval(session=mon_sess) % 20 == 0):
                #     with open('./log_folder/' + self.job_name + '_log', 'a') as f:
                #         f.write('\nstep: ' + str(j) + "\tglobal_step: " +
                #                 str(self.global_step.eval(session=mon_sess)))
            with open('./log_folder/' + self.job_name + '_log', 'a') as f:
                f.write('\nstep: ' + str(j) + "\tglobal_step: " +
                        str(self.global_step.eval(session=mon_sess)))
            save_start = time.time()
            saver.save(get_session(mon_sess),
                       self.train_folder + "/latest_model_" + self.job_name + ".ckpt")
            save_end = time.time()
            self.cumulative_time_saved += save_end - save_start
            self.log_time(self.job_name, 'end', time.time() - self.cumulative_time_saved)

        with self.train_interrupt_lock:
            self.train_interrupt = False
        with self.last_trained_lock:
            self.last_trained_sample = j
            self.last_trained_job = self.job_name

        self.log(f"Finished training job {self.job_name} on samples [{lo}, {self.last_trained_sample}].")
        with self.status_lock:
            self.status = "FREE"

    def cross_validate(self, job_name="default"):

        with self.status_lock:
            self.log(f"Status: {self.status}")
            if self.status == "BUSY":
                return
            self.status = "BUSY"

        
        self.set_job(job_name)
        self.log(f"Starting cross validation for job {self.job_name}.")

        self.build_model()

        with tf.Session(target=self.server.target) as sess:
            with open('./log_folder/' + self.job_name + '_log', 'a') as f:
                f.write('\n\nevaluating the accuracy on the validation set')

            ckpt = tf.train.latest_checkpoint(self.train_folder)
            self.log(f"CHECKPOINT: {ckpt}")
            saver = tf.train.import_meta_graph(
                self.train_folder + "/latest_model_" + self.job_name + ".ckpt.meta",
                clear_devices=True)
            saver.restore(sess, ckpt)

            cv_auc_list = []
            cv_acc_list = []
            cv_loss_list = []
            for v in range(0,
                           len(self.cv_x) - int(len(self.cv_x) % validating_size),
                           validating_size):

                acc_on_cv, loss_on_cv, preds = sess.run(
                    [self.acc, self.cross_entropy,
                     tf.nn.softmax(self.y_pred)],
                    feed_dict={
                        self.x: self.cv_x[v:v + validating_size],
                        self.y_true: self.cv_y[v:v + validating_size],
                        self.hold_prob1: 1.0,
                        self.hold_prob2: 1.0
                    })

                auc_on_cv = roc_auc_score(self.cv_y[v:v + validating_size], preds)
                cv_acc_list.append(acc_on_cv)
                cv_auc_list.append(auc_on_cv)
                cv_loss_list.append(loss_on_cv)

            acc_cv_ = round(np.mean(cv_acc_list), 5)
            auc_cv_ = round(np.mean(cv_auc_list), 5)
            loss_cv_ = round(np.mean(cv_loss_list), 5)
            self.acc_list.append(acc_cv_)
            self.auc_list.append(auc_cv_)
            self.loss_list.append(loss_cv_)
            with open('./log_folder/' + self.job_name + '_log', 'a') as f:
                f.write("\nAccuracy:" + str(acc_cv_) + "\tLoss:" +
                        str(loss_cv_) + "\tAUC:" + str(auc_cv_))

        self.log("Cross validation finished.")
        with self.status_lock:
            self.status = "FREE"

    def test(self, job_name="default"):
        with self.status_lock:
            self.log(f"Status: {self.status}")
            if self.status == "BUSY":
                return
            self.status = "BUSY"

        self.set_job(job_name)
        self.log(f"Starting testing for job {self.job_name}.")

        self.build_model()

        with tf.Session(target=self.server.target) as sess:
            with open('./log_folder/' + self.job_name + '_log', 'a') as f:
                f.write('\n\ntest the model accuracy after training')

            # self.saver.restore(sess,
            #               self.train_folder + "/latest_model_" + self.job_name + ".ckpt")
            ckpt = tf.train.latest_checkpoint(self.train_folder)
            self.log(f"CHECKPOINT: {ckpt}")
            saver = tf.train.import_meta_graph(
                self.train_folder + "/latest_model_" + self.job_name + ".ckpt.meta",
                clear_devices=True)
            saver.restore(sess, ckpt)

            test_auc_list = []
            test_acc_list = []
            test_loss_list = []
            for v in range(0,
                           len(self.test_x) - int(len(self.test_x) % validating_size),
                           validating_size):

                acc_on_test, loss_on_test, preds = sess.run(
                    [self.acc, self.cross_entropy,
                     tf.nn.softmax(self.y_pred)],
                    feed_dict={
                        self.x: self.test_x[v:v + validating_size],
                        self.y_true: self.test_y[v:v + validating_size],
                        self.hold_prob1: 1.0,
                        self.hold_prob2: 1.0
                    })

                auc_on_test = roc_auc_score(self.test_y[v:v + validating_size],
                                            preds)
                test_acc_list.append(acc_on_test)
                test_auc_list.append(auc_on_test)
                test_loss_list.append(loss_on_test)

            test_acc_ = round(np.mean(test_acc_list), 5)
            test_auc_ = round(np.mean(test_auc_list), 5)
            test_loss_ = round(np.mean(test_loss_list), 5)
            with open('./log_folder/' + self.job_name + '_log', 'a') as f:
                f.write("\nTest Results are below:")
                f.write("\nAccuracy: " + str(test_acc_) + "\tLoss: " +
                        str(test_loss_) + "\tAUC: " + str(test_auc_))

            with open('./loss_folder/loss_' + self.job_name, 'w') as f:
                f.write(str(test_loss_))
            with open('./accuracy_folder/accuracy_' + self.job_name, 'w') as f:
                f.write(str(test_acc_))

        self.log(f"Finished testing for job {self.job_name}.")
        with self.status_lock:
            self.status = "FREE"

    def load_data(self):

        self.log("Loading data.")

        data_dir = os.path.join(os.getcwd(), 'datasets')
        if not os.path.exists(data_dir):
            # Then unzip.
            with zipfile.ZipFile("datasets.zip", "r") as zip_ref:
                zip_ref.extractall()

        #Reading .npy files
        train_data = np.load(
            os.path.join(data_dir, 'train_data_mc.npy'))
        test_data = np.load(
            os.path.join(data_dir, 'test_data_mc.npy'))

        #In order to implement ALEXNET, we are resizing them to (227,227,3)
        for i in range(len(train_data)):
            train_data[i][0] = cv2.resize(train_data[i][0],
                                          (IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET))

        for i in range(len(test_data)):
            test_data[i][0] = cv2.resize(test_data[i][0],
                                         (IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET))

        train = train_data[:4800]
        cv = train_data[4800:]

        self.steps = len(train)
        self.remaining = self.steps % self.step_size

        self.X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE_ALEXNET,
                                                    IMG_SIZE_ALEXNET, 3)
        self.Y = np.array([i[1] for i in train])

        self.cv_x = np.array([i[0] for i in cv]).reshape(-1, IMG_SIZE_ALEXNET,
                                                    IMG_SIZE_ALEXNET, 3)
        self.cv_y = np.array([i[1] for i in cv])
        self.test_x = np.array([i[0]
                           for i in test_data]).reshape(-1, IMG_SIZE_ALEXNET,
                                                        IMG_SIZE_ALEXNET, 3)
        self.test_y = np.array([i[1] for i in test_data])

        self.log("Data loaded.")

    def build_model(self, verbose=False):

        self.log("Building model.")

        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % self.idx,
                    cluster=self.cluster)):

            self.log(f"Server target: {self.server.target}")

            # Build model...

            #Resetting graph
            #tf.reset_default_graph()

            #Defining Placeholders
            self.x = tf.placeholder(
                tf.float32,
                shape=[None, IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3])
            self.y_true = tf.placeholder(tf.float32, shape=[None, output_classes])

            ##CONVOLUTION LAYER 1
            #Weights for layer 1
            w_1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96],
                                                  stddev=0.01))
            #Bias for layer 1
            b_1 = tf.Variable(tf.constant(0.0, shape=[[11, 11, 3, 96][3]]))
            #Applying convolution
            c_1 = tf.nn.conv2d(self.x, w_1, strides=[1, 4, 4, 1], padding='VALID')
            #Adding bias
            c_1 = c_1 + b_1
            #Applying RELU
            c_1 = tf.nn.relu(c_1)
            # if log:
            #     self.log(c_1)
            ##POOLING LAYER1
            p_1 = tf.nn.max_pool(c_1,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID')
            # if log:
            #     self.log(p_1)

            ##CONVOLUTION LAYER 2
            #Weights for layer 2
            w_2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256],
                                                  stddev=0.01))
            #Bias for layer 2
            b_2 = tf.Variable(tf.constant(1.0, shape=[[5, 5, 96, 256][3]]))
            #Applying convolution
            c_2 = tf.nn.conv2d(p_1, w_2, strides=[1, 1, 1, 1], padding='SAME')
            #Adding bias
            c_2 = c_2 + b_2
            #Applying RELU
            c_2 = tf.nn.relu(c_2)

            # if log:
            #     self.log(c_2)

            ##POOLING LAYER2
            p_2 = tf.nn.max_pool(c_2,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID')
            # if log:
            #     self.log(p_2)

            ##CONVOLUTION LAYER 3
            #Weights for layer 3
            w_3 = tf.Variable(
                tf.truncated_normal([3, 3, 256, 384], stddev=0.01))
            #Bias for layer 3
            b_3 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 256, 384][3]]))
            #Applying convolution
            c_3 = tf.nn.conv2d(p_2, w_3, strides=[1, 1, 1, 1], padding='SAME')
            #Adding bias
            c_3 = c_3 + b_3
            #Applying RELU
            c_3 = tf.nn.relu(c_3)

            # if log:
            #     self.log(c_3)

            ##CONVOLUTION LAYER 4
            #Weights for layer 4
            w_4 = tf.Variable(
                tf.truncated_normal([3, 3, 384, 384], stddev=0.01))
            #Bias for layer 4
            b_4 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 384][3]]))
            #Applying convolution
            c_4 = tf.nn.conv2d(c_3, w_4, strides=[1, 1, 1, 1], padding='SAME')
            #Adding bias
            c_4 = c_4 + b_4
            #Applying RELU
            c_4 = tf.nn.relu(c_4)

            # if log:
            #     self.log(c_4)

            ##CONVOLUTION LAYER 5
            #Weights for layer 5
            w_5 = tf.Variable(
                tf.truncated_normal([3, 3, 384, 256], stddev=0.01))
            #Bias for layer 5
            b_5 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 256][3]]))
            #Applying convolution
            c_5 = tf.nn.conv2d(c_4, w_5, strides=[1, 1, 1, 1], padding='SAME')
            #Adding bias
            c_5 = c_5 + b_5
            #Applying RELU
            c_5 = tf.nn.relu(c_5)

            # if log:
            #     self.log(c_5)

            ##POOLING LAYER3
            p_3 = tf.nn.max_pool(c_5,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID')

            # if log:
            #     self.log(p_3)

            #Flattening
            flattened = tf.reshape(p_3, [-1, 6 * 6 * 256])

            # if log:
            #     self.log(flattened)

            ##Fully Connected Layer 1
            #Getting input nodes in FC layer 1
            input_size = int(flattened.get_shape()[1])
            #Weights for FC Layer 1
            w1_fc = tf.Variable(
                tf.truncated_normal([input_size, nodes_fc1], stddev=0.01))
            #Bias for FC Layer 1
            b1_fc = tf.Variable(tf.constant(1.0, shape=[nodes_fc1]))
            #Summing Matrix calculations and bias
            s_fc1 = tf.matmul(flattened, w1_fc) + b1_fc
            #Applying RELU
            s_fc1 = tf.nn.relu(s_fc1)

            #Dropout Layer 1
            self.hold_prob1 = tf.placeholder(tf.float32)
            s_fc1 = tf.nn.dropout(s_fc1, keep_prob=self.hold_prob1)

            # if log: 
            #     self.log(s_fc1)

            ##Fully Connected Layer 2
            #Weights for FC Layer 2
            w2_fc = tf.Variable(
                tf.truncated_normal([nodes_fc1, nodes_fc2], stddev=0.01))
            #Bias for FC Layer 2
            b2_fc = tf.Variable(tf.constant(1.0, shape=[nodes_fc2]))
            #Summing Matrix calculations and bias
            s_fc2 = tf.matmul(s_fc1, w2_fc) + b2_fc
            #Applying RELU
            s_fc2 = tf.nn.relu(s_fc2)

            # if log:
            #     self.log(s_fc2)

            #Dropout Layer 2
            self.hold_prob2 = tf.placeholder(tf.float32)
            s_fc2 = tf.nn.dropout(s_fc2, keep_prob=self.hold_prob1)

            ##Fully Connected Layer 3
            #Weights for FC Layer 3
            w3_fc = tf.Variable(
                tf.truncated_normal([nodes_fc2, output_classes], stddev=0.01))
            #Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
            b3_fc = tf.Variable(tf.constant(1.0, shape=[output_classes]))
            #Summing Matrix calculations and bias
            self.y_pred = tf.matmul(s_fc2, w3_fc) + b3_fc
            #Applying RELU

            if verbose:
                self.log(c_1)
                self.log(p_1)
                self.log(c_2)
                self.log(p_2)
                self.log(c_3)
                self.log(c_4)
                self.log(c_5)
                self.log(p_3)
                self.log(flattened)
                self.log(s_fc1)
                self.log(s_fc2)
                self.log(self.y_pred)

            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true,
                                                           logits=self.y_pred))

            self.global_step = tf.train.get_or_create_global_step()

            self.train_op = tf.train.AdamOptimizer(0.00001).minimize(
                self.cross_entropy, global_step=self.global_step)

            matches = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))
            self.acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            self.log("Model built.")

    def log(self, s):
        with self.print_lock:
            print(f"===== WORKER_{self.idx}:", end=" ")
            print(s)

    def log_time(self, job_name, action, time):
        if action != 'start' and action != 'end':
            raise Exception(f"Invalid action: {action}")
        with open(f'./log_folder/worker_{self.idx}_log', 'a') as f:
            f.write(f'{job_name},{action},{time}\n')

    def set_job(self, job_name):
        ### Move this to be a function called in train, cross_validate, and test
        # We want one worker to be able to execute a task from any job.
        self.job_name = job_name
        # train_dir = os.path.join(os.getcwd(), "checkpoints", self.job_name)
        train_dir = os.path.join(os.getcwd(), f"{self.job_name}_checkpoints")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        self.train_folder = train_dir

    def reset(self):
        self.status = "FREE"
        self.train_interrupt = False
        self.last_trained_job = None
        self.last_trained_sample = None
        self.acc_list = []
        self.auc_list = []
        self.loss_list = []

        log_folder = './log_folder'
        for file in os.listdir(log_folder):
            path = os.path.join(log_folder, file)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except Exception as e:
                self.log(e)

    def receive(self, message):

        tokens = message.split()
        command = tokens[0]

        if command == "TRAIN":
            job_name = tokens[1]
            i = int(tokens[2])
            self.train(job_name, i)
            status = None
            with self.status_lock:
                status = self.status
            return f"{status}"
        elif command == "TRAIN_INTERRUPT":
            with self.status_lock:
                if self.status != "BUSY":
                    return "FALSE"
                with self.train_interrupt_lock:
                    self.train_interrupt = True
                self.log(f"Suspending training of {self.job_name}.")
                self.status = "STOPPING"
            return "TRUE"
        elif command == "VALIDATE":
            job_name = tokens[1]
            self.cross_validate(job_name)
            status = None
            with self.status_lock:
                status = self.status
            return f"{status}"
        elif command == "TEST":
            job_name = tokens[1]
            self.test(job_name)
            status = None
            with self.status_lock:
                status = self.status
            return f"{status}"
        elif command == "RESET":
            self.reset()
            return "TRUE"
        else: # POLL
            status = None
            last_trained_job = None
            last_trained_sample = None
            with self.status_lock:
                status = self.status
            with self.last_trained_lock:
                last_trained_job = self.last_trained_job
                last_trained_sample = self.last_trained_sample
            return f"{status} {last_trained_job} {last_trained_sample}"

    def handleClient(self, connection):
        while True:
            # try:
            data = connection.recv(1024)
            if data:
                # Set the response to echo back the recieved data 
                req = data.decode()
                # self.log(req)
                response = self.receive(req)
                connection.send(response.encode())
                return True
            else:
                raise Exception('Client disconnected')
            # except Exception as e:
            #     self.log(f"Unexpected error: {e}")
            #     connection.close()
            #     return False

    def listen(self, verbose=False):
        self.log(f"Listening on port {self.port}")
        while True:
            connection, address = self.sock.accept()
            if verbose:
                self.log(f'Worker connected by {address} at {time.ctime()}')
            Thread(target = self.handleClient, args = (connection,)).start()

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_ps', type=int, required=True, help='number of parameter servers')
    parser.add_argument('--num_workers', type=int, required=True, help='number of workers')
    parser.add_argument('--worker_index', type=int, required=True, help='worker index')
    parser.add_argument('--listen', help='whether or not to run the socket server', action='store_true')
    parser.add_argument('--train', help='whether to train or not', action='store_true')
    parser.add_argument('--cross_validate', help='whether to validate or not', action='store_true')
    parser.add_argument('--test', help='whether to validate or not', action='store_true')
    parser.add_argument("--train_start", default=0, type=int, help="start index of training")
    parser.add_argument("--train_end", default=None, type=int, help="end index of training")

    args = parser.parse_args()
    print(args)

    print(f"{args.num_ps} ps and {args.num_workers} workers.")
    cluster = get_cluster(args.num_ps, args.num_workers)
    print(cluster)

    worker = Worker('', get_worker_port(args.worker_index), cluster, args.worker_index)

    if args.listen:
        worker.listen()
    if args.train:
        worker.train("default", args.train_start, args.train_end)
    if args.cross_validate:
        worker.cross_validate()
    if args.test:
        worker.test()

if __name__ == "__main__":
    main()