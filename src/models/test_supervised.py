import argparse
import tensorflow as tf
import wandb

import sys

# import local helpers
sys.path.append('/workspace/app/src')

from data.dataset_helper import *
import params

def run_model(num_classes, projectid, runid, batch_size=64):

    # https://wandb.ai/cal-capstone/bigearthnet_classification/runs/3hrh6yok/overview?workspace=user-taeil
    # model = tf.keras.models.load_model("wandb/run-20210220_210154-3hrh6yok/files/model-best.h5")
    # '$PROJECT_NAME/$RUN_ID'
    saved_model = wandb.restore('model-best.h5', run_path=f"{projectid}/{runid}", replace=False)

    model = tf.keras.models.load_model(saved_model.name)

    test_data = get_batched_dataset(params.TEST_FILENAMES, batch_size=batch_size, shuffle=False,
                                    num_classes=num_classes)
    test_steps = params.TEST_SIZE // batch_size

    perf = model.evaluate(test_data, batch_size=batch_size, steps=test_steps, return_dict=False)

    print(perf)

    wandb.run.summary["test_loss"] = perf[0]
    wandb.run.summary["test_tp"] = perf[1]
    wandb.run.summary["test_fp"] = perf[2]
    wandb.run.summary["test_tn"] = perf[3]
    wandb.run.summary["test_fn"] = perf[4]
    wandb.run.summary["test_accuracy"] = perf[5]
    wandb.run.summary["test_precision"] = perf[6]
    wandb.run.summary["test_recall"] = perf[7]
    wandb.run.summary["test_auc"] = perf[8]
    # wandb.run.summary["test_tfa_f1"] = perf[9]
    # wandb.run.summary["test_tfa_f05"] = perf[10]
    # wandb.run.summary["test_tfa_f2"] = perf[10]
    # wandb.run.summary["test_tfa_f6"] = perf[12]

    if (perf[6] + perf[7]) == 0:
        wandb.run.summary["test_f1"] = 0
    else:
        wandb.run.summary["test_f1"] = 2 * perf[6] * perf[7] / (perf[6] + perf[7])

if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help="batch size to use during test")
    parser.add_argument('-p', '--project', default="bigearthnet_classification", type=str,
                        help="project ID")
    parser.add_argument('-r', '--run', type=str,
                        help="run ID")
    parser.add_argument('-c', '--classes', default="1", type=int,
                        help="number of classes. 1 or 43")
    args = parser.parse_args()

    wandb.init(project=args.project, entity="cal-capstone")
    wandb.config.update(args)  # adds all of the arguments as config variables
    wandb.config.update({'framework': f'TensorFlow {tf.__version__}'})

    run_model(num_classes=args.classes,
              projectid=args.project,
              runid=args.run,
              batch_size=args.batch_size)
