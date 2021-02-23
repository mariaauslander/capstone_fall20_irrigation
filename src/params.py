import os
import tensorflow as tf

IMBALANCED_TRAIN_SIZE = 269695
IMBALANCED_VAL__SIZE  = 123723
IMBALANCED_TEST_SIZE = 125866

BALANCED_TRAIN_SIZE = 13932
BALANCED_VAL_SIZE = 6606
# BALANCED_TEST_SIZE = 6604

DOWNSAMPLED_TRAIN_SIZE = 69660
DOWNSAMPLED_VAL_SIZE = 33030

BALANCED_VY_TRAIN_SIZE = 9790
BALANCED_VY_VAL_SIZE = 4758
# BALANCED_VY_TEST_SIZE = 4500

BALANCED_TRAINING_FILENAMES = 'balanced_train-part-*.tfrecord'
BALANCED_VALIDATION_FILENAMES = 'balanced_val-part-*.tfrecord'
BALANCED_TEST_FILENAMES = 'balanced_test-part-*.tfrecord'

DOWNSAMPLED_TRAINING_FILENAMES = 'balanced_10_90_train-part-*.tfrecord'
DOWNSAMPLED_VALIDATION_FILENAMES = 'balanced_10_90_val-part-*.tfrecord'
DOWNSAMPLED_TEST_FILENAMES = 'balanced_10_90_test-part-*.tfrecord'

IMBALANCED_TRAINING_FILENAMES = 'train-part-*.tfrecord'
IMBALANCED_VALIDATION_FILENAMES = 'val-part-*.tfrecord'
IMBALANCED_TEST_FILENAMES = 'test-part-*.tfrecord'
IMBALANCED_CA_TRAINING_FILENAMES = 'train_ca_part*.tfrecord'

BASE_PATH = '/workspace/app'
# OUTPUT_PATH = os.path.join(BASE_PATH, 'models/supervised')
TFR_PATH = os.path.join(BASE_PATH, 'data/processed')

TEST_FILENAMES = os.path.join(TFR_PATH, "original", IMBALANCED_TEST_FILENAMES)
TEST_SIZE = IMBALANCED_TEST_SIZE

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    # tfa.metrics.F1Score(name='tfa_f1', num_classes=43),
    # tfa.metrics.FBetaScore(name='tfa_f05', num_classes=1, beta=0.5),
    # tfa.metrics.FBetaScore(name='tfa_f2', num_classes=43),    # tfa.metrics.FBetaScore(name='tfa_f2', num_classes=1, beta=2.0), beta=2.0),
    # tfa.metrics.FBetaScore(name='tfa_f6', num_classes=1, beta=6.0)
]


