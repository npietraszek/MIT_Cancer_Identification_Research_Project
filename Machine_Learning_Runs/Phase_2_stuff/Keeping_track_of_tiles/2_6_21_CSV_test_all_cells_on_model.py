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


# Code imported from internet
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#from keras import backend as K
#K.set_image_dim_ordering('tf')

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback



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


import os, sys
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)


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


#    for root, dirs, files in os.walk(
#            r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro"):
#        for the_dir in dirs:
def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)

r'''
def find_original_cell_ROI_old(cell_folder_name):

    #path_parts = splitall(cell_folder_name)

    #the_folder = path_parts[-2]
    #print(the_folder)
    final_folder = r""

    for path in Path(r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro").rglob("*.tif"):
        #print(path)
        path_parts = splitall(path)
        the_name = os.path.splitext(path_parts[-1])[0]
        if the_name == cell_folder_name:
            final_folder = os.path.split(path)[0]
            #print("the cell folder is " + str(cell_folder_name))
            #print("the final folder is " + str(final_folder))
            the_folders = os.path.join(path_parts[-7],path_parts[-6],path_parts[-5],path_parts[-4],path_parts[-3],path_parts[-2])
            the_path = os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro", the_folders)
            new_path = os.path.join(directory_to_move_to, the_folders)
            #checkDirectory(new_path)
            try:
                shutil.copytree(final_folder, new_path)
            except FileNotFoundError as e:
                print("FileNotFoundError: %s : %s" % (new_path, e.strerror))
            except FileExistsError as e:
                print("FileNotFoundError: %s : %s" % (new_path, e.strerror))
            return new_path

    print("Not found")
'''


def find_original_cell_ROI(cell_folder_name, device_folder_name, macroPath):

    #path_parts = splitall(cell_folder_name)

    #the_folder = path_parts[-2]
    #print(the_folder)
    final_folder = r""
    path_to_use = ""
    if device_folder_name == "device 1 chip 1 and 2":
        path_to_use = os.path.join(macroPath, "device 1", "chip 1 and 2")
    elif device_folder_name == "device 1 chip 3":
        path_to_use = os.path.join(macroPath, "device 1", "chip 3")
    elif device_folder_name == "device 2":
        path_to_use = os.path.join(macroPath, "device 2")
    elif device_folder_name == "device 3 chip 1 2 3":
        path_to_use = os.path.join(macroPath, "device 3 chip 1 2 3","40kDa__0001")
    elif device_folder_name == "device 3 chip 3":
        path_to_use = os.path.join(macroPath, "device 3 chip 3", "ROIs")
    else:
        raise ValueError("Path not found.")
    print(path_to_use)
    for path in Path(path_to_use).rglob("*.tif"):
        #print(path)
        path_parts = splitall(path)
        the_name = os.path.splitext(path_parts[-1])[0]
        if the_name == cell_folder_name:
            final_folder = os.path.split(path)[0]
            return final_folder

    raise ValueError("Final folder not found.")

r'''
tfback._get_available_gpus = _get_available_gpus

shape_aux= (3,20,50,50)
model = md.new_20X_model_4(input_shape=shape_aux)
model.summary()

nb_epochs = 100
batch_sz=16
loss_function='categorical_crossentropy'


adam=optimizers.Adam(clipvalue=1)
model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge


int_eph=0
csv_logger = CSVLogger(filename)
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [lrate, checkpoint,csv_logger]

config.gpu_options.allow_growth = True

final_testing_data = []
final_testing_labels = []
final_testing_directories = []
starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\20X images check V17 (macro V3 no standard)\step5 4D matrices"
directory_to_move_to = r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro - Copy"

for root, dirs, files in os.walk(starting_directory):
    for dir in dirs:
        final_testing_data.append(os.path.join(starting_directory, dir, "Final_5D_array.npy"))
        final_testing_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))
        final_testing_directories.append(dir)



number_testing_examples = 9527

model.load_weights(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Machine_Learning\Weights\9_19_20_new_run_1_(model_4,50-90-harsh_bal(20-30-30),0.001_learning,rotate)\Superloop_run_0", r"9_19_20_Test1_balanced_0.001learning_DNN.30.hdf5"))
correct_counter = 0
wrong_cancer_cells = 0
wrong_fibroblasts = 0
all_cells_list = []
cells_that_were_wrong_list = []
cells_that_were_wrong_fibroblasts = []
cells_that_were_wrong_cancer_cells = []
# Testing examples * 4 due to the rotated matrices...
for i in range(number_testing_examples):
    is_correct = False
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


    print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_testing_data[i])
    the_new_pathing = find_original_cell_ROI(final_testing_directories[i])
    #print(directory_to_write_to_csv)
    #csv_getter(the_new_pathing,model.predict(current_example))
#test_acc = model.evaluate_generator(generator = test_gen, verbose=1)
#print('\nTest loss:', test_acc)

model.summary()
print("The maximum accuracy was " + str(correct_counter))
#      + " with testing loss " + str(test_acc))
print("Total number of wrong cancer cells = " + str(wrong_cancer_cells))
print("Total number of wrong fibroblasts = " + str(wrong_fibroblasts))
#print("List of wrong cells: " + str(cells_that_were_wrong_list))
#print("Total number of epochs = "+ str(nb_epochs))

#print("Model running is +" + )

# print("Model running is +" + )
'''


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

base_folder= r'D:\MIT_Tumor_Identifcation_Project_Stuff'
#input_path = base_folder
weight_save_folder=r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\1_16_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\Superloop_run_0"
shortened_weight_save_folder = os.path.split(weight_save_folder)[0]
filepath=os.path.join(weight_save_folder,"1_16_21_Test1_balanced_0.001learning_DNN.{epoch:02d}.hdf5")
filename=os.path.join(weight_save_folder,'training.log')
checkDirectory(weight_save_folder)
epoch_name_part_1 = r"1_16_21_Test1_balanced_0.001learning_DNN."
epoch_name_part_2 = r".hdf5"

shape_aux = (3, 20, 50, 50)
model = md.new_20X_model_4(input_shape=shape_aux)
list_of_accuracy_values = []
list_of_testing_loss = []

epoch_part_model = r"1_16_21_Test1_balanced_0.001learning_DNN.0.hdf5"

# Must read datasets from text file
directory_of_cells = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step5 4D matrices"
superloop_epoch_with_maximum_accuracy = 29
superloop_counter_maximum_accuracy = 0


def split_strip_list_conversion(initial_text):
    initial_text = initial_text.replace("'","")
    initial_text = initial_text.replace("\\\\", "\\")
    initial_text.replace("]","")
    initial_text.replace("\n", "")
    initial_text.replace("[", "")
    res = initial_text.strip('][\n').split(', ')
    return res

final_compiled_testing_data = []
final_compiled_testing_labels = []
for root, dirs, files in os.walk(directory_of_cells):
    for dir in dirs:
        final_compiled_testing_data.append(os.path.join(directory_of_cells,dir,"Final_5D_array.npy"))
        final_compiled_testing_labels.append(os.path.join(directory_of_cells,dir,"Label_matrix.npy"))


device_directory_list = []
final_testing_directories_list = []
# Find "Final testing_directories"
for the_filepath in final_compiled_testing_data:
    p = Path(the_filepath)
    print(p.parts)
    print(p.parts[7])
    x = p.parts[7]
    if "device 1 chip 1 and 2" in x:
        #dictionary_of_final_testing_directories[x[0:21]] = x[21:]
        print(x[0:21])
        device_directory_list.append(x[0:21])
        final_testing_directories_list.append(x[22:])
    elif "device 1 chip 3" in x:
        print(x[0:15])
        device_directory_list.append(x[0:15])
        final_testing_directories_list.append(x[16:])
    elif "device 2" in x:
        print(x[0:8])
        device_directory_list.append(x[0:8])
        final_testing_directories_list.append(x[9:])
    elif "device 3 chip 1 2 3" in x:
        print(x[0:19])
        device_directory_list.append(x[0:19])
        final_testing_directories_list.append(x[20:])
    elif "device 3 chip 3" in x:
        print(x[0:15])
        device_directory_list.append(x[0:15])
        final_testing_directories_list.append(x[16:])
    else:
        raise ValueError("Device number not found")
    #print(device_directory_list)
    #print(final_testing_directories_list)
    #print("a")
#device_directory_list.reverse()
#final_testing_directories_list.reverse()
#'ROI10_01.oib - Series 1-1 0000_cell10_1Fb0Tc_accuracy57.7679index20'


# Python 3 code to demonstrate
# removing duplicated from list
# using naive methods



number_validation_examples = 100
number_testing_examples = 100
nb_epochs = 40
batch_sz=16
loss_function='categorical_crossentropy'


adam=optimizers.Adam(clipvalue=1)
model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge

test_gen = nicholas_generator(final_compiled_testing_data,final_compiled_testing_labels,batch_sz)


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

model.load_weights(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Best Models\1_16_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\Superloop 0",r"1_16_21_Test1_balanced_0.001learning_DNN.29.hdf5"))
correct_counter = 0
# Testing examples * 4 due to the rotated matrices...
for i in range(len(final_compiled_testing_data)):
    is_correct = False
    current_example = np.expand_dims((np.array(np.load(final_compiled_testing_data[i]))), axis=0)
    current_prediction = model.predict(current_example)
    testing_answer = np.array(np.load(final_compiled_testing_labels[i]))
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

    print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_compiled_testing_data[i])
    print(final_testing_directories_list[i])

    the_new_pathing = find_original_cell_ROI(final_testing_directories_list[i], device_directory_list[i], r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V5 Macro - Copy")
    print("The path found was " + str(the_new_pathing))
    csv_getter(the_new_pathing,model.predict(current_example))

test_acc = model.evaluate_generator(generator=test_gen, verbose=0)
print('\nTest loss:', test_acc)

model.summary()
print("The maximum accuracy was " + str(correct_counter)
      + " with testing loss " + str(test_acc))
print("Total number of epochs = " + str(nb_epochs))
list_of_accuracy_values.append(correct_counter)
list_of_testing_loss.append(test_acc)
