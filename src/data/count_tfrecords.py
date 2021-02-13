#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tensorflow as tf

print(pd.__version__)
print(tf.__version__)

tf_file_path_list = ['balanced_train.tfrecord', 'balanced_test.tfrecord','balanced_val.tfrecord',
                     'balanced_vy_train.tfrecord', 'balanced_vy_test.tfrecord', 'balanced_vy_val.tfrecord'
                     ]
for csv_file in tf_file_path_list:
    tf_records_filename = '/workspace/app/data/processed/'+csv_file
    c = 0
    for record in tf.python.python_io.tf_record_iterator(tf_records_filename):
        c += 1
    print (csv_file,c)