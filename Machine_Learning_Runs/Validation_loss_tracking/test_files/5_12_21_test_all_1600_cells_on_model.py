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

#base_folder= r'D:\MIT_Tumor_Identifcation_Project_Stuff'
#input_path = base_folder
#weight_save_folder=r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\3_25_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\Superloop_run_4"
#shortened_weight_save_folder = os.path.split(weight_save_folder)[0]
#starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_without_1600"
#saved_1600_cells_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_1600_cells"
#filepath=os.path.join(weight_save_folder,"3_25_21_Test1_balanced_0.001learning_DNN.{epoch:02d}.hdf5")
#filename=os.path.join(weight_save_folder,'training.log')
#checkDirectory(weight_save_folder)


shape_aux = (3, 20, 50, 50)
model = md.new_20X_model_4(input_shape=shape_aux)
list_of_accuracy_values = []
list_of_testing_loss = []


# Must read datasets from text file
file1 = open(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\Half_Training_All_Images\11_19_21_Test2_(half_training)\Superloop_run_0","Datasets.txt"),"r")
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

# No shuffling required

# We must remove the duplicates from this list so that the testing is only on the original cells from this long list
# such that the csv files are not overwritten

# initializing list
indices_list = []
print("The original list has " + str(len(final_saved_testing_data)) + " entries.")


no_duplicates_compiled_testing_data = []
no_duplicates_compiled_testing_labels = []
for i in range(len(final_saved_testing_data)):
    if os.path.split(final_saved_testing_data[i])[0] not in no_duplicates_compiled_testing_data:
        no_duplicates_compiled_testing_data.append(os.path.split(final_saved_testing_data[i])[0])
        no_duplicates_compiled_testing_labels.append(os.path.split(final_saved_testing_labels[i])[0])
    else:
        indices_list.append(i)

    # printing list after removal
print("The list after removing duplicates length data " + str(len(no_duplicates_compiled_testing_data)))
print("The list after removing duplicates length labels " + str(len(no_duplicates_compiled_testing_labels)))
final_compiled_testing_data = []
final_compiled_testing_labels = []
for path_index in range(len(no_duplicates_compiled_testing_data)):
    final_compiled_testing_data.append(os.path.join(no_duplicates_compiled_testing_data[path_index],"Final_5D_array_1.npy"))
    final_compiled_testing_data.append(
        os.path.join(no_duplicates_compiled_testing_data[path_index], "Final_5D_array_2.npy"))
    final_compiled_testing_data.append(
        os.path.join(no_duplicates_compiled_testing_data[path_index], "Final_5D_array_3.npy"))
    final_compiled_testing_data.append(
        os.path.join(no_duplicates_compiled_testing_data[path_index], "Final_5D_array_4.npy"))
    final_compiled_testing_labels.append(os.path.join(no_duplicates_compiled_testing_labels[path_index],"Label_matrix.npy"))
    final_compiled_testing_labels.append(
        os.path.join(no_duplicates_compiled_testing_labels[path_index], "Label_matrix.npy"))
    final_compiled_testing_labels.append(
        os.path.join(no_duplicates_compiled_testing_labels[path_index], "Label_matrix.npy"))
    final_compiled_testing_labels.append(
        os.path.join(no_duplicates_compiled_testing_labels[path_index], "Label_matrix.npy"))

print(final_compiled_testing_data[10])
print(final_compiled_testing_labels[10])
device_directory_list = []
final_testing_directories_list = []


# Find "Final testing_directories"
for the_filepath in final_compiled_testing_data:
    p = Path(the_filepath)
    print(p.parts)
    print(p.parts[5])
    x = p.parts[5]
    if "device 1 chip 1 and 2" in x:
        #dictionary_of_final_testing_directories[x[0:21]] = x[21:]
        device_directory_list.append(x[0:21])
        final_testing_directories_list.append(x[22:])
    elif "device 1 chip 3" in x:
        device_directory_list.append(x[0:15])
        final_testing_directories_list.append(x[16:])
    elif "device 2" in x:
        device_directory_list.append(x[0:8])
        final_testing_directories_list.append(x[9:])
    elif "device 3 chip 1 2 3" in x:
        device_directory_list.append(x[0:19])
        final_testing_directories_list.append(x[20:])
    elif "device 3 chip 3" in x:
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

train_gen = nicholas_generator(final_training_data,final_training_labels,batch_sz)
valid_gen = nicholas_generator(final_validation_data,final_validation_labels,batch_sz)
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


num_fibroblasts = 0
num_cancer_cells = 0
for i in range(len(final_compiled_testing_labels)):
    testing_answer = np.array(np.load(final_compiled_testing_labels[i]))
    if testing_answer[0] == 1:
        num_fibroblasts += 1
    elif testing_answer[1] == 1:
        num_cancer_cells += 1

model.load_weights(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\Half_Training_All_Images\11_19_21_Test2_(half_training)\Superloop_run_0",r"11_19_21_Test2_DNN.11.hdf5"))
correct_counter = 0
# Testing examples * 4 due to the rotated matrices...

fibroblasts_correct = 0
cancer_correct = 0
fibroblasts_wrong = 0
cancer_wrong = 0

for i in range(len(final_compiled_testing_data)):
    is_correct = False
    current_example = np.expand_dims((np.array(np.load(final_compiled_testing_data[i]))), axis=0)
    current_prediction = model.predict(current_example)
    testing_answer = np.array(np.load(final_compiled_testing_labels[i]))
    # need to account for the fact that current predictions don't always amount to 1.
    # might want to use the current prediction index instead
    '''
    if testing_answer[0] == 1 and current_prediction[0][0] == 1:
        # actually a fibroblast, predicted a fibroblast
        fibroblasts_correct += 1
    elif testing_answer[0] == 1 and current_prediction[0][1] == 1:
        # actually a fibroblast, predicted a cancer cell
        fibroblasts_wrong += 1
    elif testing_answer[1] == 1 and current_prediction[0][1] == 1:
        cancer_correct += 1
    elif testing_answer[1] == 1 and current_prediction[0][0] == 1:
        cancer_wrong += 1
    '''
    current_prediction_index = 0
    testing_answer_index = 0
    for x in range(2):
        if current_prediction[0][x] == max(current_prediction[0]):
            current_prediction_index = x
        if testing_answer[x] == max(testing_answer):
            testing_answer_index = x
    if current_prediction_index == testing_answer_index:
        if current_prediction_index == 0:
            fibroblasts_correct += 1
        elif current_prediction_index == 1:
            cancer_correct += 1
        correct_counter += 1
        is_correct = True
    else:
        if current_prediction_index == 0:
            fibroblasts_wrong += 1
        elif current_prediction_index == 1:
            cancer_wrong += 1

    print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_compiled_testing_data[i])
    print(final_testing_directories_list[i])
    '''
    the_new_pathing = find_original_cell_ROI(final_testing_directories_list[i], device_directory_list[i], r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Reconstruction\new 20x batch Testing with Tuan's V5 Macro")
    print("The path found was " + str(the_new_pathing))
    csv_getter(the_new_pathing,model.predict(current_example))
    '''

test_acc = model.evaluate_generator(generator=test_gen, verbose=0)
print('\nTest loss:', test_acc)

model.summary()
print("The maximum accuracy was " + str(correct_counter)
      + " with testing loss " + str(test_acc))
print("Percentage accuracy is " + str(100 * correct_counter / len(final_compiled_testing_data)) + "%")
print("Total number of epochs = " + str(nb_epochs))
print("Cancer cells wrong = ", cancer_wrong)
print("Cancer cells right = ", cancer_correct)
print("Fibroblasts wrong = ", fibroblasts_wrong)
print("Fibroblasts right = ", fibroblasts_correct)
print("Number of fibroblasts is", num_fibroblasts)
print("Number of cancer cells is", num_cancer_cells)
'''
header = ['1600 cell set accuracy', '1600 cell accuracy percentage', 'Minimum validation loss epoch number', 'country_code3']
data = ['Afghanistan', 652090, 'AF', 'AFG']

with open('countries.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerow(data)
'''
list_of_accuracy_values.append(correct_counter)
list_of_testing_loss.append(test_acc)


#14