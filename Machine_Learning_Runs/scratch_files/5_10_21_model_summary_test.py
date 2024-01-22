import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import numpy as np
# import tensorflow as tf
import math
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.utils import Sequence
from random import shuffle
import random
import re
import sys
# sys.path is a list of absolute path strings
sys.path.append("")


from Common_Utils.machine_learning.models.nicholas_models_phase_2_new_testing import new_20X_model_12

# Code imported from internet
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#from keras import backend as K
#K.set_image_dim_ordering('tf')

shape_aux = (3, 20, 50, 50)
model = new_20X_model_12(input_shape=shape_aux)
model.summary()