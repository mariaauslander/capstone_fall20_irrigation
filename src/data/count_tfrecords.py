#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tensorflow as tf

print(pd.__version__)
print(tf.__version__)


tf_records_filename = '/workspace/app/data/processed/final_balanced_val.tfrecord'
c = 0
for record in tf.python.python_io.tf_record_iterator(tf_records_filename):
    c += 1
print ("final_balanced_val.tfrecord",c)

tf_records_filename = '/workspace/app/data/processed/final_balanced_val_vy.tfrecord'
c = 0
for record in tf.python.python_io.tf_record_iterator(tf_records_filename):
    c += 1
print ("final_balanced_val_vy.tfrecord",c)

tf_records_filename = '/workspace/app/data/processed/train.tfrecord'
c = 0
for record in tf.python.python_io.tf_record_iterator(tf_records_filename):
    c += 1
print ("train.tfrecord",c)

tf_records_filename = '/workspace/app/data/processed/test.tfrecord'
c = 0
for record in tf.python.python_io.tf_record_iterator(tf_records_filename):
    c += 1
print ("test.tfrecord",c)

tf_records_filename = '/workspace/app/data/processed/val.tfrecord'
c = 0
for record in tf.python.python_io.tf_record_iterator(tf_records_filename):
    c += 1
print ("val.tfrecord",c)


