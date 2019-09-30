from sklearn.metrics import roc_auc_score
import cv2
import argparse
import sys
import os
import zipfile
import tensorflow as tf
import numpy as np

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session


step_size = 8
IMG_SIZE_ALEXNET = 227
validating_size = 40
nodes_fc1 = 4096
nodes_fc2 = 4096
output_classes = 4

TRAIN_DIR = os.getcwd()

class Worker:
    def __init__(self, queue, master_queue, cluster, worker_idx, job_name="default"):
        self.queue = queue
        self.master_queue = master_queue
        self.cluster = cluster
        self.idx = worker_idx

        self.server = tf.distribute.Server(cluster,
                         job_name="worker",
                         task_index=self.idx)

        train_dir = os.path.join(os.getcwd(), "checkpoints", job_name)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        self.train_folder = train_dir

        self.log(self.server)

        self.load_data()
        self.build_model()

        # The StopAtStepHook handles stopping after running given steps.
        self.hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        #init = tf.global_variables_initializer()
        self.acc_list = []
        self.auc_list = []
        self.loss_list = []
        self.saver = tf.train.Saver()

        # Limit to 1 thread
        self.session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

    def run(self):
        while True:
            task = self.queue.get()
            if task == "END":
                break
            self.master_queue.put(f"Worker {self.idx} has completed {task}.")

    def train(self):
        self.log("Starting training.")

        #while num_epoch < epochs:
        with tf.train.MonitoredTrainingSession(
                master=self.server.target,
                is_chief=(self.idx == 0),
                checkpoint_dir=train_folder,
                hooks=hooks,
                config=session_conf) as mon_sess:

            for j in range(0, steps - remaining, step_size):
                #Feeding step_size-amount data with 0.5 keeping probabilities on DROPOUT LAYERS
                _, c = mon_sess.run(
                    [train_op, cross_entropy],
                    feed_dict={
                        x: X[j:j + step_size],
                        y_true: Y[j:j + step_size],
                        hold_prob1: 0.5,
                        hold_prob2: 0.5
                    })

                if (global_step.eval(session=mon_sess) % 20 == 0):
                    with open('./log_folder/' + j_name + '_log', 'a') as f:
                        f.write('\nstep: ' + str(j) + "\tglobal_step: " +
                                str(global_step.eval(session=mon_sess)))

            saver.save(get_session(mon_sess),
                       train_folder + "/latest_model_" + j_name + ".ckpt")

        self.log("Training finished.")

    def cross_validate(self):
        pass

    def test(self):
        pass

    def load_data(self):

        self.log("Loading data.")

        #Reading .npy files
        train_data = np.load(
            os.path.join(os.getcwd(), 'datasets', 'train_data_mc.npy'))
        test_data = np.load(
            os.path.join(os.getcwd(), 'datasets', 'test_data_mc.npy'))

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
        self.remaining = self.steps % step_size

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

    def build_model(self):

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

            self.log(c_1)
            ##POOLING LAYER1
            p_1 = tf.nn.max_pool(c_1,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID')
            self.log(p_1)

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

            self.log(c_2)

            ##POOLING LAYER2
            p_2 = tf.nn.max_pool(c_2,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID')
            self.log(p_2)

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

            self.log(c_3)

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

            self.log(c_4)

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

            self.log(c_5)

            ##POOLING LAYER3
            p_3 = tf.nn.max_pool(c_5,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID')
            self.log(p_3)

            #Flattening
            flattened = tf.reshape(p_3, [-1, 6 * 6 * 256])
            self.log(flattened)

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
            hold_prob1 = tf.placeholder(tf.float32)
            s_fc1 = tf.nn.dropout(s_fc1, keep_prob=hold_prob1)

            self.log(s_fc1)

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
            self.log(s_fc2)

            #Dropout Layer 2
            hold_prob2 = tf.placeholder(tf.float32)
            s_fc2 = tf.nn.dropout(s_fc2, keep_prob=hold_prob1)

            ##Fully Connected Layer 3
            #Weights for FC Layer 3
            w3_fc = tf.Variable(
                tf.truncated_normal([nodes_fc2, output_classes], stddev=0.01))
            #Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
            b3_fc = tf.Variable(tf.constant(1.0, shape=[output_classes]))
            #Summing Matrix calculations and bias
            self.y_pred = tf.matmul(s_fc2, w3_fc) + b3_fc
            #Applying RELU
            self.log(self.y_pred)

            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true,
                                                           logits=self.y_pred))

            self.global_step = tf.train.get_or_create_global_step()

            self.train_op = tf.train.AdagradOptimizer(0.00001).minimize(
                self.cross_entropy, global_step=self.global_step)

            matches = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))
            self.acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            self.log("Model built.")

    def log(self, s):
        print(f"===== WORKER_{self.idx}:", end=" ")
        print(s)