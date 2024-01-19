'''
This test reveals that CSV files cannot be overwritten with a CSV writer by default.
Must either delete CSVs manually or create a function that overwrites them to replace CSV files
'''

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
import shutil
from pathlib import Path
from MIT_Tumor_Identifcation_Project.Machine_learning_runs.Phase_2_stuff import nicholas_models_phase_2_new_testing as md
import csv

def csv_getter(filename,prediction):
    # field names
    fields = ['Fibroblast','Cancer']

    # data rows of csv file.
    current_prediction_index = 0
    for x in range(2):
        if prediction[0][x] == max(prediction[0]):
            current_prediction_index = x


    if current_prediction_index == 1:
        # Cancer
        rows = [[0,1]]
    else:
        # Fibroblast
        rows = [[1,0]]
    # name of csv file
    full_filename = os.path.join(filename,"the_csv_file.csv")
    # writing to csv file
    with open(full_filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

csv_getter(r"D:\MIT_Tumor_Identifcation_Project_Stuff\CSV Data Test",[[1,0]])
