from flags import FLAGS
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
import os

class utils():
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.data_dir = FLAGS.data_dir
        self.train_data = os.path.join(self.data_dir, 'train.csv')
        self.test_data = os.path.join(self.data_dir, 'test.csv')
        self.val_data = os.path.join(self.data_dir, 'train.csv')
        self.xv_size = self.get_xv_size()

    def get_xv_size(self):
        return 784

    def get_train_batch(self):
        df = pd.read_csv(self.train_data)
        num = len(df.index)
        #print num
        #print df.head()
        while(True):
            df = df.sample(frac=1)
            #print df.head()
            begin = 0
            while begin < num:
              if begin + self.batch_size < num:
                end = begin + self.batch_size
              else:
                end = num
              x = np.array(df.iloc[begin : end, 1:]).astype(float)
              y = np.array([i for i in df.iloc[begin : end, 0]])
              yield x, y
              begin += self.batch_size

    def get_test_batch(self, mode='test'):
        if mode == 'test':
          df = pd.read_csv(self.test_data)
        else:
          df = pd.read_csv(self.val_data) 
        num = len(df.index)
        begin = 0
        while begin < num:
          if begin + self.batch_size < num:
            end = begin + self.batch_size
          else:
            end = num
          if mode == 'test':
            x = np.array(df.iloc[begin : end, :]).astype(float)
            yield x
          else:
            x = np.array(df.iloc[begin : end, 1:]).astype(float)
            y = np.array([i for i in df.iloc[begin : end, 0]])
            yield x, y
          begin += self.batch_size

def write_test(id, pre):
    cf = csv.writer(open(os.path.join(FLAGS.data_dir, 'prediction.csv'), 'w'))
    cf.writerow(['ImageId', 'Label'])
    for i, p in zip(id, pre):
      cf.writerow([i, p])

def check_data():
  step = 0
  for x, y in utils().get_train_batch():
    step += 1
    if step > 100:
      break

if __name__ == '__main__':
  check_data()

