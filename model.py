import tensorflow as tf
from utils import utils, write_test
import os, csv

class Model():
    def __init__(self, args):
        self.mode = args.mode
        self.model_dir = args.model_dir
        self.model_path = os.path.join(self.model_dir, 'model')
        self.data_dir = args.data_dir
        self.model_type = args.model_type
        self.units = args.units
        self.filter = args.filter
        self.kernel = args.kernel
        self.load = args.load
        self.print_step = args.print_step
        self.save_step = args.save_step
        self.max_step = args.max_step

        self.utils = utils()
        self.xv_size = self.utils.xv_size
        self.dp = args.dp

        self.sess = tf.Session()
        self.build(self.model_type)
        self.saver = tf.train.Saver(max_to_keep = 10)

    def build(self, model_type):
        self.x = tf.placeholder(tf.float32, shape=[None, self.xv_size])
        self.y = tf.placeholder(tf.int64, shape=[None])
        self.y_ = tf.one_hot(self.y, 10)

        if model_type == 'dnn':
          dense = tf.layers.dense(self.x, self.units[0], activation=tf.nn.relu)
          z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
          for i in range(1, len(self.units)):
            dense = tf.layers.dense(z, self.units[i], activation=tf.nn.relu)
            z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
        elif model_type == 'cnn':
          x_cnn = tf.reshape(self.x, [-1, 28, 28, 1])
          conv = tf.layers.conv2d(x_cnn, filters=self.filter[0], kernel_size = self.kernel, padding='same', activation=tf.nn.relu)
          pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
          for i in range(1, len(self.filter)):
            conv = tf.layers.conv2d(pool, filters=self.filter[i], kernel_size = self.kernel, padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
          pool = tf.reshape(pool, [-1, 7 * 7 * self.filter[-1]])
          dense = tf.layers.dense(pool, 1024, activation=tf.nn.relu)
          z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
          
        self.logits = tf.layers.dense(z, 10)

        # train
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))
        self.op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        #self.op = tf.train.MomentumOptimizer(0.001, 0.8).minimize(self.loss)

        # test
        self.pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        #print self.pred.shape
        #print self.y.shape
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))

    def train(self):

        self.sess.run(tf.global_variables_initializer()) 
        step = 1
        acc = 0.0
        loss = 0.0

        if not os.path.exists(self.model_dir):
            os.system("mkdir -p {}".format(self.model_dir))
        
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))
            step = int(ckpt.model_checkpoint_path.split('-')[-1])

        for x, y in self.utils.get_train_batch():
            feed_dict = {self.x : x, self.y : y}
            output = [self.op, self.acc, self.loss]
            _, acc_temp, loss_temp = self.sess.run(output, feed_dict)

            acc += acc_temp
            loss += loss_temp

            if step % self.print_step == 0:
                acc /= self.print_step
                loss /= self.print_step
                print("Step : {}    Acc : {}    Loss : {}".format(step, acc, loss))
                acc = 0.0
                loss = 0.0

            if step % self.save_step == 0:
                print("Saving model ...")
                self.saver.save(self.sess, self.model_path, global_step=step)

            if step >= self.max_step:
                break

            step += 1

    def test(self):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        ids = []
        pre = []

        for x in self.utils.get_test_batch():
            feed_dict = {self.x : x}
            pred = self.sess.run([self.pred], feed_dict)
            pre.extend(pred[0])

        write_test([i for i in range(1, len(pre)+1)], pre)

    def val(self):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        acc = 0.0
        cnt = 0

        for x, y in self.utils.get_test_batch(mode='val'):
            feed_dict = {self.x : x, self.y : y}
            acc_temp = self.sess.run([self.acc], feed_dict)
            cnt += len(x)
            acc += acc_temp[0] * len(x)
        acc /= float(cnt)
        print("Acc : {}".format(acc))

