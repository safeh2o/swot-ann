import sys
from swotann.swot_ml import SWOT_ML
#import tensorflow as tf
#tf.keras.utils.set_random_seed(0)
import random
import numpy as np
import tensorflow as tf
import keras
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def run_swot():
    SWOT_net = SWOT_ML()
    input_file = sys.argv[1]
    results_file = sys.argv[3]
    report_file = sys.argv[4]
    storage_target = sys.argv[5]
    SWOT_net.run_swot(input_file, results_file, report_file, storage_target)


if __name__ == "__main__":
    run_swot()
