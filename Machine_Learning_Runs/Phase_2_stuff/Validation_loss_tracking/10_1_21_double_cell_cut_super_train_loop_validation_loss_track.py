'''
3/19/21
Full machine learning run program. Runs a "superloop" of multiple machine learning epoch runs.
Each superloop iteration has a different randomization of training dataset and validation dataset and testing dataset,
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
import re

from MIT_Tumor_Identifcation_Project.Machine_learning_runs.Phase_2_stuff import nicholas_models_phase_2_new_testing as md

# Code imported from internet
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#from keras import backend as K
#K.set_image_dim_ordering('tf')
def step_decay_modify_6(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def step_decay_modify_5(epoch):
    initial_lrate = 0.001
    drop = 0.33
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def step_decay_modify_4(epoch):
    initial_lrate = 0.00001
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def step_decay_modify_3(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def step_decay_modify_2(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

'''def step_decay_modify(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    if epoch < 10:
        return initial_lrate
    else:
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
'''
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


def shuffle_matrices_list(labels_list,the_dictionary):
    combined_rotated_list = []
    combined_labels_list = []
    rotate_copy_counter = 0
    for key in the_dictionary:
        rotate_copy_counter = rotate_copy_counter + 1
        for value in the_dictionary[key]:
            combined_rotated_list.append(value)
    for y in range(rotate_copy_counter):
        for x in labels_list:
            combined_labels_list.append(x)
    data_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(combined_rotated_list)))
    shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(combined_rotated_list[i])
        labels_shuf.append(combined_labels_list[i])
    #print(data_shuf)
    #print(labels_shuf)
    return data_shuf, labels_shuf

def get_the_rotated_files():
    training_data_dict = {}
    validation_data_dict = {}
    testing_data_dict = {}
    saved_testing_data_dict = {}
    total_directory_list = []
    saved_total_directory_list = []

    training_labels = []
    validation_labels = []
    testing_labels = []
    saved_testing_labels = []
    for x in range(1, 5):
        training_data_dict["training_data_{0}".format(x)] = []
        validation_data_dict["validation_data_{0}".format(x)] = []
        testing_data_dict["testing_data_{0}".format(x)] = []
        saved_testing_data_dict["saved_testing_data_{0}".format(x)] = []



    validation_cancer_cell_counter = 1
    validation_fibroblast_counter = 1
    testing_cancer_cell_counter = 1
    testing_fibroblast_counter = 1
    training_cancer_cell_counter = 1
    training_fibroblast_counter = 1
    testing_saving_fibroblast_counter = 1
    testing_saving_cancer_counter = 1


    number_validation_cancer_cells = 50
    number_validation_fibroblasts = 50
    number_testing_cancer_cells = 50
    number_testing_fibroblasts = 50

    number_training_cancer_cells = 312
    number_training_fibroblasts = 227
    training_counter = 1
    validation_counter = 1
    testing_counter = 1
    testing_saving_counter = 1
    number_training_examples = 537
    number_validation_examples = 20
    number_testing_examples = 20

    # In theory, there should be only be around 400 unique cells in this directory. Make sure that's the case...
    list_of_directories_to_walk = next(os.walk(saved_1600_cells_directory))[1]

    saved_total_directory_list = []
    for the_directory in list_of_directories_to_walk:
        full_directory = os.path.join(saved_1600_cells_directory, the_directory)
        for root, dirs, files in os.walk(full_directory):
            for dir in dirs:
                saved_total_directory_list.append(os.path.join(full_directory, dir))

    shuffle(saved_total_directory_list)
    for the_dir in saved_total_directory_list:
        the_label_matrix = np.load(os.path.join(the_dir, "Label_matrix.npy"))

        if the_label_matrix[0] == 0:
            # We are dealing with a cancer cell
            for x in range(1, 5):
                saved_testing_data_dict["saved_testing_data_{0}".format(x)].append(
                    os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                # print(validation_data_dict["validation_data_{0}".format(x)][0])
                # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
            saved_testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

            testing_saving_cancer_counter = testing_saving_cancer_counter + 1
            testing_saving_counter = testing_saving_counter + 1

        else:
            # is_cancer_cell = False
            for x in range(1, 5):
                saved_testing_data_dict["saved_testing_data_{0}".format(x)].append(
                    os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                # print(validation_data_dict["validation_data_{0}".format(x)][0])
                # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
            saved_testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

            testing_saving_fibroblast_counter = testing_saving_fibroblast_counter + 1
            testing_saving_counter = testing_saving_counter + 1

    print("testing_saving_fibroblast_counter = " + str(testing_saving_fibroblast_counter))
    print("testing_saving_cancer_counter = " + str(testing_saving_cancer_counter))
    print("testing_saving_counter = " + str(testing_saving_counter))



    # Now on to the rest of the machine learning training, validation, and testing cells...
    list_of_directories_to_walk = next(os.walk(starting_directory))[1]

    total_directory_list = []
    for the_directory in list_of_directories_to_walk:
        full_directory = os.path.join(starting_directory, the_directory)
        for root, dirs, files in os.walk(full_directory):
            for dir in dirs:
                total_directory_list.append(os.path.join(full_directory, dir))

    print(len(total_directory_list))
    shuffle(total_directory_list)
    for the_dir in total_directory_list:
        the_label_matrix = np.load(os.path.join(the_dir, "Label_matrix.npy"))

        if the_label_matrix[0] == 0:
            # We are dealing with a cancer cell
            # If there aren't enough cancer cells for validation, add this cell to it.
            if validation_cancer_cell_counter <= number_validation_cancer_cells:
                for x in range(1, 5):
                    validation_data_dict["validation_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    #print(validation_data_dict["validation_data_{0}".format(x)][0])
                    #the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
                validation_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                validation_cancer_cell_counter = validation_cancer_cell_counter + 1
                validation_counter = validation_counter + 1
            elif testing_cancer_cell_counter <= number_testing_cancer_cells:
                for x in range(1, 5):
                    testing_data_dict["testing_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                testing_cancer_cell_counter = testing_cancer_cell_counter + 1
                testing_counter = testing_counter + 1
            #elif training_cancer_cell_counter <= number_training_cancer_cells:
            else:
                for x in range(1, 5):
                    training_data_dict["training_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                training_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                training_cancer_cell_counter = training_cancer_cell_counter + 1
                training_counter = training_counter + 1
        else:
            # is_cancer_cell = False
            # If there aren't enough fibroblasts for validation, add this cell to it.
            if validation_fibroblast_counter <= number_validation_fibroblasts:
                for x in range(1, 5):
                    validation_data_dict["validation_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    # print(validation_data_dict["validation_data_{0}".format(x)][0])
                    # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
                validation_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                validation_fibroblast_counter = validation_fibroblast_counter + 1
                validation_counter = validation_counter + 1

            elif testing_fibroblast_counter <= number_testing_fibroblasts:
                for x in range(1, 5):
                    testing_data_dict["testing_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                testing_fibroblast_counter = testing_fibroblast_counter + 1
                testing_counter = testing_counter + 1

            #elif training_fibroblast_counter <= number_training_fibroblasts:
            else:
                for x in range(1, 5):
                    training_data_dict["training_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                training_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                training_fibroblast_counter = training_fibroblast_counter + 1
                training_counter = training_counter + 1
    print("training_fibroblast_counter = " + str(training_fibroblast_counter))
    print("training_cancer_counter = " + str(training_cancer_cell_counter))
    print("training_counter = " + str(training_counter))
    print("validation_fibroblast_counter = " + str(validation_fibroblast_counter))
    print("validation_cancer_counter = " + str(validation_cancer_cell_counter))
    print("validation_counter = " + str(validation_counter))
    print("testing_fibroblast_counter = " + str(testing_fibroblast_counter))
    print("testing_cancer_counter = " + str(testing_cancer_cell_counter))
    print("testing_counter = " + str(testing_counter))
    print("testing_saving_fibroblast_counter = " + str(testing_saving_fibroblast_counter))
    print("testing_saving_cancer_counter = " + str(testing_saving_cancer_counter))
    print("testing_saving_counter = " + str(testing_saving_counter))

    '''
    # testing values for dictionaries and labels
    counter = 0
    for y in training_data_dict:
        for x in range(len(training_data_dict[y])):
            print(str(y) + " with value " + str(training_data_dict[y][x]))
            print("Corresponding label = " + training_labels[x])
    for y in validation_data_dict:
        for x in range(len(validation_data_dict[y])):
            print(str(y) + " with value " + str(validation_data_dict[y][x]))
            print("Corresponding label = " + validation_labels[x])
    for y in testing_data_dict:
        for x in range(len(testing_data_dict[y])):
            print(str(y) + " with value " + str(testing_data_dict[y][x]))
            print("Corresponding label = " + testing_labels[x])
    '''
    shuffled_training_data, shuffled_training_labels = shuffle_matrices_list(training_labels,training_data_dict)
    shuffled_validation_data, shuffled_validation_labels = shuffle_matrices_list(validation_labels,validation_data_dict)
    shuffled_testing_data, shuffled_testing_labels = shuffle_matrices_list(testing_labels,testing_data_dict)
    shuffled_saved_testing_data, shuffled_saved_testing_labels = shuffle_matrices_list(saved_testing_labels,saved_testing_data_dict)

    return shuffled_training_data,shuffled_training_labels,shuffled_validation_data,shuffled_validation_labels, shuffled_testing_data, shuffled_testing_labels, shuffled_saved_testing_data, shuffled_saved_testing_labels

def get_the_rotated_files_old():
    training_data_dict = {}
    validation_data_dict = {}
    testing_data_dict = {}
    training_labels = []
    validation_labels = []
    testing_labels = []
    for x in range(1, 5):
        training_data_dict["training_data_{0}".format(x)] = []
        validation_data_dict["validation_data_{0}".format(x)] = []
        testing_data_dict["testing_data_{0}".format(x)] = []


    validation_cancer_cell_counter = 1
    validation_fibroblast_counter = 1
    testing_cancer_cell_counter = 1
    testing_fibroblast_counter = 1
    training_cancer_cell_counter = 1
    training_fibroblast_counter = 1

    number_validation_cancer_cells = 50
    number_validation_fibroblasts = 50
    number_testing_cancer_cells = 50
    number_testing_fibroblasts = 50
    number_training_cancer_cells = 312
    number_training_fibroblasts = 227
    training_counter = 1
    validation_counter = 1
    testing_counter = 1
    number_training_examples = 537
    number_validation_examples = 20
    number_testing_examples = 20

    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            the_label_matrix = np.load(os.path.join(starting_directory, dir, "Label_matrix.npy"))
            if the_label_matrix[0] == 0:
                # We are dealing with a cancer cell

                # If there aren't enough cancer cells for validation, add this cell to it.
                if validation_cancer_cell_counter <= number_validation_cancer_cells:
                    for x in range(1, 5):
                        validation_data_dict["validation_data_{0}".format(x)].append(
                            os.path.join(starting_directory, dir, "Final_5D_array_{0}".format(x) + ".npy"))
                        #print(validation_data_dict["validation_data_{0}".format(x)][0])
                        #the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
                    validation_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))

                    validation_cancer_cell_counter = validation_cancer_cell_counter + 1
                    validation_counter = validation_counter + 1
                elif testing_cancer_cell_counter <= number_testing_cancer_cells:
                    for x in range(1, 5):
                        testing_data_dict["testing_data_{0}".format(x)].append(
                            os.path.join(starting_directory, dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    testing_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))

                    testing_cancer_cell_counter = testing_cancer_cell_counter + 1
                    testing_counter = testing_counter + 1
                #elif training_cancer_cell_counter <= number_training_cancer_cells:
                else:
                    for x in range(1, 5):
                        training_data_dict["training_data_{0}".format(x)].append(
                            os.path.join(starting_directory, dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    training_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))

                    training_cancer_cell_counter = training_cancer_cell_counter + 1
                    training_counter = training_counter + 1
            else:
                # is_cancer_cell = False

                # If there aren't enough fibroblasts for validation, add this cell to it.
                if validation_fibroblast_counter <= number_validation_fibroblasts:
                    for x in range(1, 5):
                        validation_data_dict["validation_data_{0}".format(x)].append(
                            os.path.join(starting_directory, dir, "Final_5D_array_{0}".format(x) + ".npy"))
                        # print(validation_data_dict["validation_data_{0}".format(x)][0])
                        # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
                    validation_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))

                    validation_fibroblast_counter = validation_fibroblast_counter + 1
                    validation_counter = validation_counter + 1

                elif testing_fibroblast_counter <= number_testing_fibroblasts:
                    for x in range(1, 5):
                        testing_data_dict["testing_data_{0}".format(x)].append(
                            os.path.join(starting_directory, dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    testing_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))

                    testing_fibroblast_counter = testing_fibroblast_counter + 1
                    testing_counter = testing_counter + 1

                #elif training_fibroblast_counter <= number_training_fibroblasts:
                else:
                    for x in range(1, 5):
                        training_data_dict["training_data_{0}".format(x)].append(
                            os.path.join(starting_directory, dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    training_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))

                    training_fibroblast_counter = training_fibroblast_counter + 1
                    training_counter = training_counter + 1
    print("Validation counter = " + str(validation_counter))
    print("Testing counter = " + str(testing_counter))
    print("Training counter = " + str(training_counter))

    '''
    # testing values for dictionaries and labels
    counter = 0
    for y in training_data_dict:
        for x in range(len(training_data_dict[y])):
            print(str(y) + " with value " + str(training_data_dict[y][x]))
            print("Corresponding label = " + training_labels[x])
    for y in validation_data_dict:
        for x in range(len(validation_data_dict[y])):
            print(str(y) + " with value " + str(validation_data_dict[y][x]))
            print("Corresponding label = " + validation_labels[x])
    for y in testing_data_dict:
        for x in range(len(testing_data_dict[y])):
            print(str(y) + " with value " + str(testing_data_dict[y][x]))
            print("Corresponding label = " + testing_labels[x])
    '''
    shuffled_training_data, shuffled_training_labels = shuffle_matrices_list(training_labels,training_data_dict)
    shuffled_validation_data, shuffled_validation_labels = shuffle_matrices_list(validation_labels,validation_data_dict)
    shuffled_testing_data, shuffled_testing_labels = shuffle_matrices_list(testing_labels,testing_data_dict)

    return shuffled_training_data,shuffled_training_labels,shuffled_validation_data,shuffled_validation_labels, shuffled_testing_data, shuffled_testing_labels

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

def read_validation_loss_from_dataset_log(epoch_number, the_filepath):
    print("Reading validation from " + str(epoch_number))
    file1 = open((the_filepath), "r")
    datasets = file1.readlines()
    the_line = datasets[epoch_number+1]
    print("The line is " + str(the_line))
    split_line = re.split("(,)", the_line)
    print("The split line is " + str(split_line))
    epoch_validation_loss = float(split_line[6][0:-1])
    print(epoch_validation_loss)
    return epoch_validation_loss



superloop_rounds = 1


# MUST FIGURE OUT HOW TO SAVE THE WEIGHTS IN DIFFERENT PLACES
superloop_maximum_accuracy = 0
superloop_validation_loss_with_maximum_accuracy = 0
superloop_epoch_with_maximum_validation_accuracy = 0
superloop_counter_maximum_validation_accuracy = 0

superloop_minimum_validation_loss = 100000
superloop_accuracy_with_minimum_validation_loss = 0
superloop_epoch_with_minimum_validation_loss = 0
superloop_counter_minimum_validation_loss = 0


weight_save_folder = r""
shortened_weight_save_folder = r""
epoch_name_part_1 = r""
epoch_name_part_2 = r""
filepath = r""
# Define what the model is here
shape_aux = (1, 20, 50, 50)
model = md.new_20X_model_4(input_shape=shape_aux)
model.summary()

# test_number = 15
for superloop_counter in range(superloop_rounds):
    tfback._get_available_gpus = _get_available_gpus

    base_folder= r'D:\MIT_Tumor_Identifcation_Project_Stuff'
    #input_path = base_folder
    weight_save_folder=r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\only_DAPI\10_2_21_test25_(only_DAPI)\Superloop_run_{0}".format(superloop_counter)
    shortened_weight_save_folder = os.path.split(weight_save_folder)[0]
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\Only_DAPI_without_1600"
    saved_1600_cells_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\Only_DAPI_1600_cells"
    filepath=os.path.join(weight_save_folder,"10_2_21_test25_DNN.{epoch:02d}.hdf5")
    filename=os.path.join(weight_save_folder,'training.log')
    text_results_file_name = "10_2_21_test25_only_DAPI_rotated_1600_set.txt"
    model_name_to_use = "new_20X_model_4"
    step_decay_thing = "step_decay_modify_2"
    checkDirectory(weight_save_folder)

    epoch_name_part_1 = r"10_2_21_test25_DNN."
    epoch_name_part_2 = r".hdf5"




    #number_training_examples = 10728
    number_validation_examples = 100
    number_testing_examples = 100
    nb_epochs = 30
    batch_sz=16
    loss_function='categorical_crossentropy'


    adam=optimizers.Adam(clipvalue=1)
    model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge


    int_eph=0
    csv_logger = CSVLogger(filename)
    lrate = LearningRateScheduler(step_decay_modify_2)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    callbacks_list = [lrate, checkpoint,csv_logger]

    final_training_data, final_training_labels, final_validation_data, final_validation_labels, final_testing_data, \
    final_testing_labels, final_saved_testing_data, final_saved_testing_labels = get_the_rotated_files()

    file1 = open(os.path.join(weight_save_folder, r"Datasets.txt"), "a")  # append mode
    file1.write("Datasets for this epoch weights \n")
    file1.write(str(final_training_data) + "\n") # 1
    file1.write(str(final_training_labels) + "\n") # 2
    file1.write(str(final_validation_data) + "\n") # 3
    file1.write(str(final_validation_labels) + "\n") # 4
    file1.write(str(final_testing_data) + "\n") # 5
    file1.write(str(final_testing_labels) + "\n") # 6
    file1.write(str(final_saved_testing_data) + "\n") # 7
    file1.write(str(final_saved_testing_labels) + "\n") # 8
    file1.write("       ----END----")
    file1.close()


    train_gen = nicholas_generator(final_training_data,final_training_labels,batch_sz)
    valid_gen = nicholas_generator(final_validation_data,final_validation_labels,batch_sz)
    test_gen = nicholas_generator(final_testing_data,final_testing_labels,batch_sz)

    config.gpu_options.allow_growth = True
    model.fit_generator(generator = train_gen, epochs=nb_epochs, validation_data=valid_gen, callbacks=callbacks_list, shuffle="batch", initial_epoch=int_eph, verbose = 2)



    # Tracking the training accuracy for each epoch
    # maximum_accuracy = 0
    # validation_loss_with_maximum_accuracy = 0
    # epoch_with_maximum_accuracy = 0
    #
    # minimum_validation_loss = 100000
    # accuracy_with_minimum_validation_loss = 0
    # epoch_with_minimum_validation_loss = 0
    # cells_that_were_wrong_list = []
    # wrong_fibroblasts = 0
    # wrong_cancer_cells = 0
    #
    # # Tracking the training accuracy for each epoch
    #
    # # Tracking the validation accuracy for each epoch
    #
    # # Testing the test dataset for each epoch
    # for epoch_counter in range(nb_epochs):
    #     print("Current epoch is " + str(epoch_counter))
    #     if epoch_counter + 1 < 10:
    #         model.load_weights(
    #             os.path.join(weight_save_folder, epoch_name_part_1 + "0" + str(epoch_counter + 1) + epoch_name_part_2))
    #     else:
    #         model.load_weights(
    #             os.path.join(weight_save_folder, epoch_name_part_1 + str(epoch_counter + 1) + epoch_name_part_2))
    #     correct_counter = 0
    #     # Testing examples * 4 due to the rotated matrices...
    #     for i in range(number_testing_examples * 4):
    #
    #         current_example = np.expand_dims((np.array(np.load(final_testing_data[i]))), axis=0)
    #         current_prediction = model.predict(current_example)
    #         testing_answer = np.array(np.load(final_testing_labels[i]))
    #         current_prediction_index = 0
    #         testing_answer_index = 0
    #         for x in range(2):
    #             if current_prediction[0][x] == max(current_prediction[0]):
    #                 current_prediction_index = x
    #             if testing_answer[x] == max(testing_answer):
    #                 testing_answer_index = x
    #         if current_prediction_index == testing_answer_index:
    #             correct_counter += 1
    #             is_correct = True
    #         else:
    #             cells_that_were_wrong_list.append(final_testing_data[i])
    #             if testing_answer[0] == 0:
    #                 # It's a cancer cell!
    #                 wrong_cancer_cells = wrong_cancer_cells + 1
    #             else:
    #                 # It's a fibroblast!
    #                 wrong_fibroblasts = wrong_fibroblasts + 1
    #
    #         # print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_testing_data[i])
    #
    #     # test_acc = model.evaluate_generator(generator = test_gen, verbose=1)
    #     # print('\nTest loss:', test_acc)
    #
    #     # Instead of reading testing loss here, should read validation loss from the text file
    #
    #     validation_acc = read_validation_loss_from_dataset_log(epoch_counter, filename)
    #     if (correct_counter > maximum_accuracy):
    #         maximum_accuracy = correct_counter
    #         validation_loss_with_maximum_accuracy = validation_acc
    #         epoch_with_maximum_accuracy = epoch_counter + 1
    #     # CHECKPOINT: THIS IS WHERE I STOPPED MARCH 19th
    #
    #     if (minimum_validation_loss > validation_acc):
    #         minimum_validation_loss = validation_acc
    #         accuracy_with_minimum_validation_loss = correct_counter
    #         epoch_with_minimum_validation_loss = epoch_counter + 1
    #
    #     cells_that_were_wrong_list = []
    #     wrong_fibroblasts = 0
    #     wrong_cancer_cells = 0
    #     is_correct = False
    # model.summary()
    # print("The maximum accuracy was " + str(maximum_accuracy)
    #       + " with validation loss " + str(validation_loss_with_maximum_accuracy)
    #       + " at epoch " + str(epoch_with_maximum_accuracy))
    # print("The minimum validation loss was " + str(minimum_validation_loss)
    #       + " with accuracy " + str(accuracy_with_minimum_validation_loss)
    #       + " at epoch " + str(epoch_with_minimum_validation_loss))
    # print("Total number of epochs = " + str(nb_epochs))
    # print("Total number of wrong cancer cells = " + str(wrong_cancer_cells))
    # print("Total number of wrong fibroblasts = " + str(wrong_fibroblasts))
    # # print("List of wrong cells: " + str(cells_that_were_wrong_list))
    # if (maximum_accuracy > superloop_maximum_accuracy):
    #     superloop_maximum_accuracy = maximum_accuracy
    #     superloop_validation_loss_with_maximum_accuracy = validation_loss_with_maximum_accuracy
    #     superloop_epoch_with_maximum_validation_accuracy = epoch_with_maximum_accuracy
    #     superloop_counter_maximum_validation_accuracy = superloop_counter
    # if (superloop_minimum_validation_loss > minimum_validation_loss):
    #     superloop_minimum_validation_loss = minimum_validation_loss
    #     superloop_accuracy_with_minimum_validation_loss = accuracy_with_minimum_validation_loss
    #     superloop_epoch_with_minimum_validation_loss = epoch_with_minimum_validation_loss
    #     superloop_counter_minimum_validation_loss = superloop_counter
    #

    # Tracking the validation accuracy for each epoch

    # maximum_accuracy = 0
    # validation_loss_with_maximum_accuracy = 0
    # epoch_with_maximum_accuracy = 0
    #
    # minimum_validation_loss = 100000
    # accuracy_with_minimum_validation_loss = 0
    # epoch_with_minimum_validation_loss = 0
    # cells_that_were_wrong_list = []
    # wrong_fibroblasts = 0
    # wrong_cancer_cells = 0
    # for epoch_counter in range(nb_epochs):
    #     print("Current epoch is " + str(epoch_counter))
    #     if epoch_counter + 1 < 10:
    #         model.load_weights(
    #             os.path.join(weight_save_folder, epoch_name_part_1 + "0" + str(epoch_counter + 1) + epoch_name_part_2))
    #     else:
    #         model.load_weights(
    #             os.path.join(weight_save_folder, epoch_name_part_1 + str(epoch_counter + 1) + epoch_name_part_2))
    #     correct_counter = 0
    #     # Testing examples * 4 due to the rotated matrices...
    #     for i in range(number_testing_examples * 4):
    #
    #         current_example = np.expand_dims((np.array(np.load(final_testing_data[i]))), axis=0)
    #         current_prediction = model.predict(current_example)
    #         testing_answer = np.array(np.load(final_testing_labels[i]))
    #         current_prediction_index = 0
    #         testing_answer_index = 0
    #         for x in range(2):
    #             if current_prediction[0][x] == max(current_prediction[0]):
    #                 current_prediction_index = x
    #             if testing_answer[x] == max(testing_answer):
    #                 testing_answer_index = x
    #         if current_prediction_index == testing_answer_index:
    #             correct_counter += 1
    #             is_correct = True
    #         else:
    #             cells_that_were_wrong_list.append(final_testing_data[i])
    #             if testing_answer[0] == 0:
    #                 # It's a cancer cell!
    #                 wrong_cancer_cells = wrong_cancer_cells + 1
    #             else:
    #                 # It's a fibroblast!
    #                 wrong_fibroblasts = wrong_fibroblasts + 1
    #
    #         # print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_testing_data[i])
    #
    #     # test_acc = model.evaluate_generator(generator = test_gen, verbose=1)
    #     # print('\nTest loss:', test_acc)
    #
    #     # Instead of reading testing loss here, should read validation loss from the text file
    #
    #     validation_acc = read_validation_loss_from_dataset_log(epoch_counter, filename)
    #     if (correct_counter > maximum_accuracy):
    #         maximum_accuracy = correct_counter
    #         validation_loss_with_maximum_accuracy = validation_acc
    #         epoch_with_maximum_accuracy = epoch_counter + 1
    #     # CHECKPOINT: THIS IS WHERE I STOPPED MARCH 19th
    #
    #     if (minimum_validation_loss > validation_acc):
    #         minimum_validation_loss = validation_acc
    #         accuracy_with_minimum_validation_loss = correct_counter
    #         epoch_with_minimum_validation_loss = epoch_counter + 1
    #
    #     cells_that_were_wrong_list = []
    #     wrong_fibroblasts = 0
    #     wrong_cancer_cells = 0
    #     is_correct = False
    # model.summary()
    # print("The maximum accuracy was " + str(maximum_accuracy)
    #       + " with validation loss " + str(validation_loss_with_maximum_accuracy)
    #       + " at epoch " + str(epoch_with_maximum_accuracy))
    # print("The minimum validation loss was " + str(minimum_validation_loss)
    #       + " with accuracy " + str(accuracy_with_minimum_validation_loss)
    #       + " at epoch " + str(epoch_with_minimum_validation_loss))
    # print("Total number of epochs = " + str(nb_epochs))
    # print("Total number of wrong cancer cells = " + str(wrong_cancer_cells))
    # print("Total number of wrong fibroblasts = " + str(wrong_fibroblasts))
    # # print("List of wrong cells: " + str(cells_that_were_wrong_list))
    # if (maximum_accuracy > superloop_maximum_accuracy):
    #     superloop_maximum_accuracy = maximum_accuracy
    #     superloop_validation_loss_with_maximum_accuracy = validation_loss_with_maximum_accuracy
    #     superloop_epoch_with_maximum_validation_accuracy = epoch_with_maximum_accuracy
    #     superloop_counter_maximum_validation_accuracy = superloop_counter
    # if (superloop_minimum_validation_loss > minimum_validation_loss):
    #     superloop_minimum_validation_loss = minimum_validation_loss
    #     superloop_accuracy_with_minimum_validation_loss = accuracy_with_minimum_validation_loss
    #     superloop_epoch_with_minimum_validation_loss = epoch_with_minimum_validation_loss
    #     superloop_counter_minimum_validation_loss = superloop_counter
    #

    # Function to loop over all of the epoch weight savings and record the maximum accuracy and testing loss,
    # as well as the epochs responsible for the high scores.
    # Testing the test dataset for each epoch
    print(filepath)
    maximum_accuracy = 0
    validation_loss_with_maximum_accuracy = 0
    epoch_with_maximum_accuracy = 0

    minimum_validation_loss = 100000
    accuracy_with_minimum_validation_loss = 0
    epoch_with_minimum_validation_loss = 0
    cells_that_were_wrong_list = []
    wrong_fibroblasts = 0
    wrong_cancer_cells = 0

    for epoch_counter in range(nb_epochs):
        print("Current epoch is " + str(epoch_counter))
        if epoch_counter + 1 < 10:
            model.load_weights(os.path.join(weight_save_folder,epoch_name_part_1 + "0" + str(epoch_counter + 1) + epoch_name_part_2))
        else:
            model.load_weights(os.path.join(weight_save_folder, epoch_name_part_1 + str(epoch_counter + 1) + epoch_name_part_2))
        correct_counter = 0
        # Testing examples * 4 due to the rotated matrices...
        for i in range(number_testing_examples*4):

            current_example = np.expand_dims((np.array(np.load(final_testing_data[i]))),axis = 0)
            current_prediction = model.predict(current_example)
            testing_answer = np.array(np.load(final_testing_labels[i]))
            current_prediction_index = 0
            testing_answer_index = 0
            for x in range(2):
                if current_prediction[0][x] == max(current_prediction[0]):
                    current_prediction_index = x
                if testing_answer[x] == max(testing_answer):
                    testing_answer_index = x
            if current_prediction_index == testing_answer_index:
                correct_counter += 1
                is_correct = True
            else:
                cells_that_were_wrong_list.append(final_testing_data[i])
                if testing_answer[0] == 0:
                    # It's a cancer cell!
                    wrong_cancer_cells = wrong_cancer_cells + 1
                else:
                    # It's a fibroblast!
                    wrong_fibroblasts = wrong_fibroblasts + 1

            #print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_testing_data[i])

        validation_acc = read_validation_loss_from_dataset_log(epoch_counter, filename)
        if(correct_counter > maximum_accuracy):
            maximum_accuracy = correct_counter
            validation_loss_with_maximum_accuracy = validation_acc
            epoch_with_maximum_accuracy = epoch_counter + 1

        if(minimum_validation_loss > validation_acc):
            minimum_validation_loss = validation_acc
            accuracy_with_minimum_validation_loss = correct_counter
            epoch_with_minimum_validation_loss = epoch_counter + 1

        cells_that_were_wrong_list = []
        wrong_fibroblasts = 0
        wrong_cancer_cells = 0
        is_correct = False
    model.summary()
    print("The maximum accuracy was " + str(maximum_accuracy)
          + " with validation loss " + str(validation_loss_with_maximum_accuracy)
           + " at epoch " + str(epoch_with_maximum_accuracy))
    print("The minimum validation loss was " + str(minimum_validation_loss)
          + " with accuracy " + str(accuracy_with_minimum_validation_loss)
           + " at epoch " + str(epoch_with_minimum_validation_loss))
    print("Total number of epochs = "+ str(nb_epochs))
    print("Total number of wrong cancer cells = " + str(wrong_cancer_cells))
    print("Total number of wrong fibroblasts = " + str(wrong_fibroblasts))
    #print("List of wrong cells: " + str(cells_that_were_wrong_list))
    if (maximum_accuracy > superloop_maximum_accuracy):
        superloop_maximum_accuracy = maximum_accuracy
        superloop_validation_loss_with_maximum_accuracy = validation_loss_with_maximum_accuracy
        superloop_epoch_with_maximum_validation_accuracy = epoch_with_maximum_accuracy
        superloop_counter_maximum_validation_accuracy = superloop_counter
    if (superloop_minimum_validation_loss > minimum_validation_loss):
        superloop_minimum_validation_loss = minimum_validation_loss
        superloop_accuracy_with_minimum_validation_loss = accuracy_with_minimum_validation_loss
        superloop_epoch_with_minimum_validation_loss = epoch_with_minimum_validation_loss
        superloop_counter_minimum_validation_loss = superloop_counter
    #print("Model running is +" + )
    print("The superloop maximum accuracy was " + str(superloop_maximum_accuracy)
          + " with validation loss " + str(superloop_validation_loss_with_maximum_accuracy)
          + " at epoch " + str(superloop_epoch_with_maximum_validation_accuracy)
          + " at superloop loop counter " + str(superloop_counter_maximum_validation_accuracy))
    print("The superloop minimum validation loss was " + str(superloop_minimum_validation_loss)
          + " with accuracy " + str(superloop_accuracy_with_minimum_validation_loss)
          + " at epoch " + str(superloop_epoch_with_minimum_validation_loss)
          + " at superloop loop counter " + str(superloop_counter_minimum_validation_loss))
    print("Total number of epochs across superloop = " + str(nb_epochs))

    # Test the 95% confidence interval for every superloop

    # NEEDS TESTING
    if epoch_with_minimum_validation_loss < 10:
        model.load_weights(
            os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter)), epoch_name_part_1 + "0" + str(epoch_with_minimum_validation_loss) + epoch_name_part_2))
    else:
        model.load_weights(
            os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter)), epoch_name_part_1 + str(epoch_with_minimum_validation_loss) + epoch_name_part_2))

    correct_counter = 0
    for i in range(len(final_saved_testing_data)):
        is_correct = False
        #print("Current example is " + str(final_saved_testing_data[i]))
        #debug_test = np.array(np.load(final_saved_testing_data[i]))
        current_example = np.expand_dims((np.array(np.load(final_saved_testing_data[i]))), axis=0)
        current_prediction = model.predict(current_example)
        testing_answer = np.array(np.load(final_saved_testing_labels[i]))
        current_prediction_index = 0
        testing_answer_index = 0
        for x in range(2):
            if current_prediction[0][x] == max(current_prediction[0]):
                current_prediction_index = x
            if testing_answer[x] == max(testing_answer):
                testing_answer_index = x
        if current_prediction_index == testing_answer_index:
            correct_counter += 1
            is_correct = True

        #print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_saved_testing_data[i])
    print("On all the final_saved_testing_data, the machine learning got " + str(correct_counter) + " correct out of " + str(len(final_saved_testing_data))
          + " with accuracy " + str(correct_counter / len(final_saved_testing_data) * 100) + "%")
    #     '''
    #    the_new_pathing = find_original_cell_ROI(final_testing_directories_list[i], device_directory_list[i], r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Reconstruction\new 20x batch Testing with Tuan's V5 Macro")
    #    print("The path found was " + str(the_new_pathing))
    #    csv_getter(the_new_pathing,model.predict(current_example))
    #    '''

    #final_saved_testing_data

list_of_accuracy_values = []
list_of_testing_loss = []

# Must read datasets from text file
file1 = open(os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_maximum_validation_accuracy)), r"Datasets.txt"), "r")
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

compiled_testing_data = final_testing_data + final_saved_testing_data
compiled_testing_labels = final_testing_labels + final_saved_testing_labels

for x in range(100):
    final_training_zip = list(zip(final_training_data, final_training_labels))
    random.shuffle(final_training_zip)
    final_training_data, final_training_labels = zip(*final_training_zip)

    final_validation_zip = list(zip(final_validation_data, final_validation_labels))
    random.shuffle(final_validation_zip)
    final_validation_data, final_validation_labels = zip(*final_validation_zip)

    compiled_testing_zip = list(zip(compiled_testing_data, compiled_testing_labels))
    random.shuffle(compiled_testing_zip)
    compiled_testing_data, compiled_testing_labels = zip(*compiled_testing_zip)

    number_validation_examples = 100
    number_testing_examples = 100
    nb_epochs = 40
    batch_sz=16
    loss_function='categorical_crossentropy'


    adam=optimizers.Adam(clipvalue=1)
    model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge

    train_gen = nicholas_generator(final_training_data,final_training_labels,batch_sz)
    valid_gen = nicholas_generator(final_validation_data,final_validation_labels,batch_sz)
    test_gen = nicholas_generator(compiled_testing_data[0:400],compiled_testing_labels[0:400],batch_sz)


    config.gpu_options.allow_growth = True
    # model.fit_generator(generator = train_gen, epochs=nb_epochs, validation_data=valid_gen, callbacks=callbacks_list, shuffle="batch", initial_epoch=int_eph, verbose = 2)


    # Function to loop over all of the epoch weight savings and record the maximum accuracy and testing loss,
    # as well as the epochs responsible for the high scores.

    maximum_accuracy = 0
    testing_loss_with_maximum_accuracy = 0
    epoch_with_maximum_accuracy = 0

    minimum_testing_loss = 100000
    accuracy_with_minimum_testing_loss = 0
    epoch_with_minimum_testing_loss = 0

    if superloop_epoch_with_maximum_validation_accuracy < 10:
        model.load_weights(
            os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_minimum_validation_loss)), epoch_name_part_1 + "0" + str(superloop_epoch_with_maximum_validation_accuracy) + epoch_name_part_2))
    else:
        model.load_weights(
            os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_minimum_validation_loss)), epoch_name_part_1 + str(superloop_epoch_with_maximum_validation_accuracy) + epoch_name_part_2))
    correct_counter = 0
    # Testing examples * 4 due to the rotated matrices...
    for i in range(400):
        is_correct = False
        current_example = np.expand_dims((np.array(np.load(compiled_testing_data[i]))), axis=0)
        current_prediction = model.predict(current_example)
        testing_answer = np.array(np.load(compiled_testing_labels[i]))
        current_prediction_index = 0
        testing_answer_index = 0
        for ii in range(2):
            if current_prediction[0][ii] == max(current_prediction[0]):
                current_prediction_index = ii
            if testing_answer[ii] == max(testing_answer):
                testing_answer_index = ii
        if current_prediction_index == testing_answer_index:
            correct_counter += 1
            is_correct = True

        #print(model.predict(current_example), testing_answer, is_correct, correct_counter, compiled_testing_data[i])
    test_acc = model.evaluate_generator(generator=test_gen, verbose=0)
    print('\nTest loss:', test_acc)

    # model.summary()
    print("Test number " + str(x))
    print("The maximum accuracy was " + str(correct_counter)
          + " with testing loss " + str(test_acc))
    print("Total number of epochs = " + str(nb_epochs))
    list_of_accuracy_values.append(correct_counter)
    list_of_testing_loss.append(test_acc)

max_accuracy = max(list_of_accuracy_values)
max_testing_loss = max(list_of_testing_loss)
min_testing_loss = min(list_of_testing_loss)
min_accuracy = min(list_of_accuracy_values)

mean_accuracy = np.mean(list_of_accuracy_values)
std_accuracy = np.std(list_of_accuracy_values)
mean_loss = np.mean(list_of_testing_loss)
std_loss = np.std(list_of_testing_loss)

directory_to_write_to = r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\only_DAPI"

checkDirectory(directory_to_write_to)

file1 = open(os.path.join(directory_to_write_to,text_results_file_name),"a")#append mode
file1.write("10_2_21_test_results_25   \n")
file1.write("Step decay used " + step_decay_thing)
file1.write("list_of_accuracy_values = " + str(list_of_accuracy_values) + "\n")
file1.write("Minimum_accuracy = " + str(min_accuracy) + "\n")
file1.write("Maximum_accuracy = " + str(max_accuracy) + "\n")
file1.write("Mean_accuracy = " + str(mean_accuracy) + "\n")
file1.write("Standard_deviation_accuracy = " + str(std_accuracy) + "\n")
file1.write("list_of_testing_loss = " + str(list_of_testing_loss) + "\n")
file1.write("Maximum_testing_loss = " + str(max_testing_loss) + "\n")
file1.write("Minimum_testing_loss = " + str(min_testing_loss) + "\n")
file1.write("Mean_testing_loss = " + str(mean_loss) + "\n")
file1.write("Standard_deviation_loss = " + str(std_loss) + "\n")
file1.write("Model used was " + model_name_to_use + "\n")
file1.write("The superloop maximum accuracy was " + str(superloop_maximum_accuracy)
          + " with validation loss " + str(superloop_validation_loss_with_maximum_accuracy)
          + " at epoch " + str(superloop_epoch_with_maximum_validation_accuracy)
          + " at superloop loop counter " + str(superloop_counter_maximum_validation_accuracy) + "\n" +
           ("The superloop minimum validation loss was " + str(superloop_minimum_validation_loss)
          + " with accuracy " + str(superloop_accuracy_with_minimum_validation_loss)
          + " at epoch " + str(superloop_epoch_with_minimum_validation_loss)
          + " at superloop loop counter " + str(superloop_counter_minimum_validation_loss)))
file1.write("       ----END---- \n \n \n")
file1.close()
