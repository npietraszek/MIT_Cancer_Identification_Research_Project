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
from keras import optimizers
from keras.utils import Sequence
from random import shuffle
import random
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.machine_learning.generators import nicholas_generator

from MIT_Tumor_Identifcation_Project.Machine_learning_runs.Phase_2_stuff import nicholas_models_phase_2_new_testing as md

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
weight_save_folder=r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\4_7_21_new_run_1_(new_20X_model_12,step_decay_modify_2)\Superloop_run_2"
shortened_weight_save_folder = os.path.split(weight_save_folder)[0]
starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_without_1600"
saved_1600_cells_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_1600_cells"
filepath=os.path.join(weight_save_folder,"4_7_21_Test1_DNN.{epoch:02d}.hdf5")
filename=os.path.join(weight_save_folder,'training.log')
checkDirectory(weight_save_folder)
epoch_name_part_1 = r"4_7_21_Test1_DNN."
epoch_name_part_2 = r".hdf5"
superloop_epoch_with_maximum_accuracy = 3
superloop_counter_maximum_accuracy = 2

shape_aux = (3, 20, 50, 50)
model = md.new_20X_model_12(input_shape=shape_aux)
list_of_accuracy_values = []
list_of_testing_loss = []

#epoch_part_model = r"3_27_21_Test1_balanced_0.001learning_DNN.15.hdf5"

# Must read datasets from text file
file1 = open(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\4_7_21_new_run_1_(new_20X_model_12,step_decay_modify_2)\Superloop_run_2","Datasets.txt"),"r")

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


final_training_zip = list(zip(final_training_data, final_training_labels))
random.shuffle(final_training_zip)
final_training_data, final_training_labels = zip(*final_training_zip)

final_validation_zip = list(zip(final_validation_data, final_validation_labels))
random.shuffle(final_validation_zip)
final_validation_data, final_validation_labels = zip(*final_validation_zip)

final_testing_zip = list(zip(final_testing_data, final_testing_labels))
random.shuffle(final_testing_zip)
final_testing_data, final_testing_labels = zip(*final_testing_zip)

number_validation_examples = 100
number_testing_examples = 100
nb_epochs = 40
batch_sz=16
loss_function='categorical_crossentropy'


adam=optimizers.Adam(clipvalue=1)
model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge

train_gen = nicholas_generator(final_training_data,final_training_labels,batch_sz)
valid_gen = nicholas_generator(final_validation_data,final_validation_labels,batch_sz)
test_gen = nicholas_generator(final_testing_data[0:400],final_testing_labels[0:400],batch_sz)


config.gpu_options.allow_growth = True
# model.fit_generator(generator = train_gen, epochs=nb_epochs, validation_data=valid_gen, callbacks=callbacks_list, shuffle="batch", initial_epoch=int_eph, verbose = 2)


# Function to loop over all of the epoch weight savings and record the maximum accuracy and testing loss,
# as well as the epochs responsible for the high scores.
'''
maximum_accuracy = 0
testing_loss_with_maximum_accuracy = 0
epoch_with_maximum_accuracy = 0

minimum_testing_loss = 100000
accuracy_with_minimum_testing_loss = 0
epoch_with_minimum_testing_loss = 0
'''
# D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\1_16_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\Superloop_run_0\1_16_21_Test1_balanced_0.001learning_DNN.29.hdf5
# D:\\MIT_Tumor_Identifcation_Project_Stuff\\Phase_2_stuff\\Machine Learning\\Weights\\1_16_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\\Superloop_run_0\\1_16_21_Test1_balanced_0.001learning_DNN.29.hdf5'
if superloop_epoch_with_maximum_accuracy + 1 < 10:
    path_to_model = os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_maximum_accuracy)), epoch_name_part_1 + "0" + str(superloop_epoch_with_maximum_accuracy) + epoch_name_part_2)
    model.load_weights(
        os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_maximum_accuracy)), epoch_name_part_1 + "0" + str(superloop_epoch_with_maximum_accuracy) + epoch_name_part_2))
else:
    path_to_model = os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_maximum_accuracy)), epoch_name_part_1 + str(superloop_epoch_with_maximum_accuracy) + epoch_name_part_2)
    model.load_weights(
        os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter_maximum_accuracy)), epoch_name_part_1 + str(superloop_epoch_with_maximum_accuracy) + epoch_name_part_2))
correct_counter = 0
# Testing examples * 4 due to the rotated matrices...
for i in range(400):
    is_correct = False
    current_example = np.expand_dims((np.array(np.load(final_testing_data[i]))), axis=0)
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

    #print(model.predict(current_example), testing_answer, is_correct, correct_counter, compiled_testing_data[i])
test_acc = model.evaluate_generator(generator=test_gen, verbose=0)
print('\nTest loss:', test_acc)

model.summary()
print("The maximum accuracy was " + str(correct_counter)
      + " with testing loss " + str(test_acc))
print("Total number of epochs = " + str(nb_epochs))
list_of_accuracy_values.append(correct_counter)
list_of_testing_loss.append(test_acc)

# max_accuracy = max(list_of_accuracy_values)
# max_testing_loss = max(list_of_testing_loss)
# min_testing_loss = min(list_of_testing_loss)
# min_accuracy = min(list_of_accuracy_values)
'''
directory_to_write_to = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Random_dataset_testing_results"

checkDirectory(directory_to_write_to)

file1 = open(os.path.join(directory_to_write_to,r"1_19_21_test_results_rotated_2000_set.txt"),"a")#append mode
file1.write("1_19_21_test_results_1 \n")
file1.write("list_of_accuracy_values = " + str(list_of_accuracy_values) + "\n")
file1.write("Minimum_accuracy = " + str(min_accuracy) + "\n")
file1.write("Maximum_accuracy = " + str(max_accuracy) + "\n")
file1.write("list_of_testing_loss = " + str(list_of_testing_loss) + "\n")
file1.write("Maximum_testing_loss = " + str(max_testing_loss) + "\n")
file1.write("Minimum_testing_loss = " + str(min_testing_loss) + "\n")
file1.write("       ----END---- \n \n \n")
file1.close()
'''
# print("Model running is +" + )
