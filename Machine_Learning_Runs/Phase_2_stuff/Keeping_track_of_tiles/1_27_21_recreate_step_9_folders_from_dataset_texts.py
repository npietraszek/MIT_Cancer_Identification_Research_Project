'''
6/29/20
The machine learning phase of the program.

Things to do:
1) Take the generated 5D matrices and split them into training, testing, and validation data
2) Figure out how to create the proper generator to return the matrices
3) Work out any dimension issues

TESTING INTERMISSION: MUST TEST GENERATOR, MATRICES, and DIMENSION STUFF
4) Configure the model we’d like to use
5) Tune hyperparameters of the neural net, (e.g. kernel size, number of training epochs)
6) Begin training runs!
7) Start testing accuracy
8) Gain an initial perspective on whether 20X images can work
9) Revise and refine the hyperparameters of the neural net



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

# Code imported from internet
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#from keras import backend as K
#K.set_image_dim_ordering('tf')
def step_decay_modify(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    if epoch < 10:
        return initial_lrate
    else:
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class LossHistory():
     def on_train_begin(self, logs={}):
        self.losses = []

     def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# Kwabena's generator
class kwabena_generator(Sequence) :

    def __init__(self, filenames, batch_size) :
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        #return np.array([np.resize(np.squeeze(np.load(file_name)[0]),(1,64,64,64)) for file_name in batch]), np.array([np.resize(np.squeeze(np.load(file_name)[1]),(1,64,64,64)) for file_name in batch])
        #return np.array([np.reshape(np.load(file_name)[0,0],(64,64,64,2)) for file_name in batch]), np.array([np.reshape(np.load(file_name)[0,1],(3,64,64,64)) for file_name in batch])
        return np.array([np.reshape(np.load(file_name)['input'],(64,64,64,3)) for file_name in batch]), np.array([np.reshape(np.load(file_name)['output']/np.sum(np.load(file_name)['output'],1),(3,64,64,64)) for file_name in batch])

# Old generator
# Generator in testing. STILL NEEDS VERIFICATION
class nicholas_generator(Sequence) :
    def __init__(self, matrix_filenames, label_filenames, batch_size) :

        self.matrix_filenames = matrix_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.matrix_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        matrix_batch = self.matrix_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        label_batch = self.label_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        #return np.array([np.resize(np.squeeze(np.load(file_name)[0]),(1,64,64,64)) for file_name in batch]), np.array([np.resize(np.squeeze(np.load(file_name)[1]),(1,64,64,64)) for file_name in batch])
        #return np.array([np.reshape(np.load(file_name)[0,0],(64,64,64,2)) for file_name in batch]), np.array([np.reshape(np.load(file_name)[0,1],(3,64,64,64)) for file_name in batch])
        return np.array([np.load(file_name) for file_name in matrix_batch]), np.array([np.load(file_name) for file_name in label_batch])

# Utility function to check if a directory exists.
def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)


import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus

starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step8 Rotated ROI sorted 4D"
saved_1600_cell_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_1600_cells"
non1600_cell_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_without_1600"
checkDirectory(starting_directory)
checkDirectory(non1600_cell_directory)
checkDirectory(saved_1600_cell_directory)



shape_aux = (3, 20, 50, 50)
model = md.new_20X_model_4(input_shape=shape_aux)
list_of_accuracy_values = []
list_of_testing_loss = []

#epoch_part_model = r"1_19_21_Test1_balanced_0.001learning_DNN.29"

# Must read datasets from text file
file1 = open(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\1_16_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\Superloop_run_0","Datasets - Copy.txt"),"r")
superloop_epoch_with_maximum_accuracy = 29
superloop_counter_maximum_accuracy = 0
datasets = file1.readlines()

def split_strip_list_conversion(initial_text):
    initial_text = initial_text.replace("'","")
    initial_text = initial_text.replace("\\\\", "\\")
    initial_text.replace("]","")
    initial_text.replace("\n", "")
    initial_text.replace("[", "")
    res = initial_text.strip('][\n').split(', ')
    return res

test_var = datasets[1]
final_training_data = split_strip_list_conversion(datasets[1])
final_training_labels = split_strip_list_conversion(datasets[2])
final_validation_data = split_strip_list_conversion(datasets[3])
final_validation_labels = split_strip_list_conversion(datasets[4])
final_testing_data = split_strip_list_conversion(datasets[5])
final_testing_labels = split_strip_list_conversion(datasets[6])
final_saved_testing_data = split_strip_list_conversion(datasets[7])
final_saved_testing_labels = split_strip_list_conversion(datasets[8])

#device_1_chip_1_and_2_count = 0

list_of_ROIS = []

# Take all the ROI folders that were inside the 1600 folder
# and extract the folder name out of the path from step 9
for the_file_path in final_saved_testing_data:
    #first_split = os.path.split(the_file_path)[0]
    #print(first_split)
    p = Path(the_file_path)
    #print(p.parts)
    #print(p.parts[7])
    if p.parts[7] not in list_of_ROIS:
        list_of_ROIS.append(p.parts[7])
print("The 1600cell list is now " + str(list_of_ROIS))
print("Length of the 1600cell list: " + str(len(list_of_ROIS)))

# Walk through and get all of them
list_of_directories_to_walk = next(os.walk(starting_directory))[1]

print("Full list of directories in step8 = " + str(list_of_directories_to_walk))
print("Length of the step8 list: " + str(len(list_of_directories_to_walk)))
# Compare this total list to the list of ROIs from the 1600 text file

for cell_1600_directory in list_of_ROIS:
    list_of_directories_to_walk.remove(cell_1600_directory)
print("Length of the step8 list after duplicate removal: " + str(len(list_of_directories_to_walk)))

# Expand the 1600cell folder names we got to full filepaths inside the step 8 folder
saved_1600_directory_list = []
for the_directory in list_of_ROIS:
    full_directory = os.path.join(starting_directory,the_directory)
    saved_1600_directory_list.append(full_directory)
    '''
    for root, dirs, files in os.walk(full_directory):
        for dir in dirs:
            saved_total_directory_list.append(os.path.join(full_directory, dir))
    '''
print("1600cell List with full file names is " + str(saved_1600_directory_list))
print("Length of the 1600cell full directory list " + str(len(saved_1600_directory_list)))
# Expand the full step8 folder names we got to full filepaths inside the step 8 folder
saved_total_directory_list = []
for the_directory in list_of_directories_to_walk:
    full_directory = os.path.join(starting_directory,the_directory)
    saved_total_directory_list.append(full_directory)
    '''
    for root, dirs, files in os.walk(full_directory):
        for dir in dirs:
            saved_total_directory_list.append(os.path.join(full_directory, dir))
    '''
print("Non1600cell List with full file names is " + str(saved_total_directory_list))
print("Length of the non_1600 full directory list " + str(len(saved_total_directory_list)))


num_cells_in_non1600_directory = 0
# Now that we have the full path names, we can copy over the entire folders over to the step 9 folders.
# However we are going to need to make new paths...
for ROI_directory in saved_total_directory_list:
    the_name_of_directory = os.path.split(ROI_directory)[1]
    #print(the_name_of_directory)
    directory_to_copy_to = os.path.join(non1600_cell_directory, the_name_of_directory)
    print(ROI_directory)
    print(directory_to_copy_to)
    for the_file in Path(ROI_directory).rglob('*.txt'):
        file1 = open(the_file, "r+")
        number_cells = file1.read()
    num_cells_in_non1600_directory = num_cells_in_non1600_directory + int(number_cells)
    print("Number cells = " + str(number_cells))
    shutil.copytree(ROI_directory, directory_to_copy_to)

print("Total number of cells in non1600 directory = " + str(num_cells_in_non1600_directory))

num_cells_in_1600_directory = 0
# Now that we have the full path names, we can copy over the entire folders over to the step 9 folders.
# However we are going to need to make new paths...
for ROI_directory in saved_1600_directory_list:
    the_name_of_directory = os.path.split(ROI_directory)[1]
    # print(the_name_of_directory)
    directory_to_copy_to = os.path.join(saved_1600_cell_directory, the_name_of_directory)
    print(ROI_directory)
    print(directory_to_copy_to)
    for the_file in Path(ROI_directory).rglob('*.txt'):
        file1 = open(the_file, "r+")
        number_cells = file1.read()
    num_cells_in_1600_directory = num_cells_in_1600_directory + int(number_cells)
    print("Number cells = " + str(number_cells))
    shutil.copytree(ROI_directory, directory_to_copy_to)

print("Total number of cells in 1600 directory = " + str(num_cells_in_1600_directory))

'''
# Create the full pathname from stuff in step 8
for root, dirs, files in next(os.walk(starting_directory)):
    for the_dir in dirs:
        x = str(the_dir)
        for the_ROI in list_of_ROIS:
            if the_ROI in x:
                print(str(the_ROI) + " found inside " + str(x))
'''
'''
# Take ROI folders from the dictionary of cells until 1600 cells are obtained in total from the dictionary
# We do 400 because each file has 4 rotated images of cells
total_number = 0
while total_number < 400:
    cell_directory, directory_cell_number = random.choice(list(dictionary_of_file_counters.items()))
    del dictionary_of_file_counters[cell_directory]
    print("Taking directory " + str(cell_directory) + " as part of the saved 1600.")
    print("Total number of cells within: " + str(directory_cell_number))
    the_name_of_directory = os.path.split(cell_directory)[1]
    shutil.copytree(cell_directory, os.path.join(saved_1600_cell_directory, the_name_of_directory))
    total_number = total_number + int(directory_cell_number)

    print("Current total number: " + str(total_number))


# Take the rest of the cells and put them into a different folder for machine learning to draw from
# for all of training, validation, and testing
for cell_key in dictionary_of_file_counters:
    the_name_of_cell_directory = os.path.split(cell_key)[1]
    print("Moving directory " + str(cell_key) + " over to the normal machine learning folder")
    shutil.copytree(cell_key,os.path.join(new_directory, the_name_of_cell_directory))

    # some house keeping to make sure we got all the cells
    total_number = total_number +  int(dictionary_of_file_counters[cell_key])

print("Total number of cells moved to both directories: " + str(total_number))
'''
#D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_without_1600\device 1 chip 3 ROI1_01.oib\\device 1 chip 3 ROI1_01.oib - Series 1-1 0014_cell3_1Fb0Tc_accuracy93.0667index12\\Final_5D_array_2.npy
#D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_1600_cells\device 1 chip 3 ROI1_01.oib\device 1 chip 3 ROI1_01.oib - Series 1-1 0014_cell3_1Fb0Tc_accuracy93.0667index12