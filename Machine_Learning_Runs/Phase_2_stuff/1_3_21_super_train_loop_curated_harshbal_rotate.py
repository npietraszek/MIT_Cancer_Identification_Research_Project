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
'''
class nicholas_generator(Sequence):
    def __init__(self,dataArray,labelArray, batch_size):
        self.dataArray = dataArray
        self.dataArraySampleNum = len(dataArray[0])
        self.labelArray = labelArray
        self.batch_size = batch_size
    def __len__(self):
        return (np.ceil(len(self.dataArray) / float(self.batch_size))).astype(np.int)
    def __getitem__(self,idx):


        reshaped_dataArray = self.dataArray
        reshaped_labelArray = self.labelArray
        x = reshaped_dataArray[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = reshaped_labelArray[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.array(x)
        y = np.array(y)
        return x, y
'''
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


'''training_data_1_shuf = []
    training_data_2_shuf = []
    training_data_3_shuf = []
    training_data_4_shuf = []
    training_labels_shuf = []
    index_shuf = list(range(len(training_data_1)))
    shuffle(index_shuf)
    for i in index_shuf:
        training_data_1_shuf.append(training_data_1[i])
        training_data_2_shuf.append(training_data_2[i])
        training_data_3_shuf.append(training_data_3[i])
        training_data_4_shuf.append(training_data_4[i])
        training_labels_shuf.append(training_labels[i])
    print(training_data_1_shuf)
    print(training_data_2_shuf)
    print(training_data_3_shuf)
    print(training_data_4_shuf)
    print(training_labels_shuf)'''
'''
data_1_shuf = []
data_2_shuf = []
data_3_shuf = []
data_4_shuf = []
labels_shuf = []
index_shuf = list(range(len(rotated_list_1)))
shuffle(index_shuf)
for i in index_shuf:
    data_1_shuf.append(rotated_list_1[i])
    data_2_shuf.append(rotated_list_2[i])
    data_3_shuf.append(rotated_list_3[i])
    data_4_shuf.append(rotated_list_4[i])
    labels_shuf.append(labels_list[i])
print(data_1_shuf)
print(data_2_shuf)
print(data_3_shuf)
print(data_4_shuf)
print(labels_shuf)
return data_1_shuf, data_2_shuf, data_3_shuf, data_4_shuf, labels_shuf
'''
'''
# Possible function to make "get_the_rotated_files" easier to read.
# Must rework this...
def shuffle_matrices_list(rotated_list_1,rotated_list_2,rotated_list_3,rotated_list_4,labels_list):
    combined_rotated_list = []
    combined_labels_list = []
    for x in rotated_list_1:
        combined_rotated_list.append(x)
    for x in rotated_list_2:
        combined_rotated_list.append(x)
    for x in rotated_list_3:
        combined_rotated_list.append(x)
    for x in rotated_list_4:
        combined_rotated_list.append(x)
    for y in range(4):
        for x in labels_list:
            combined_labels_list.append(x)
    data_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(combined_rotated_list)))
    shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(combined_rotated_list[i])
        labels_shuf.append(combined_labels_list[i])
    print(data_shuf)
    print(labels_shuf)
    return data_shuf, labels_shuf
'''
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
    total_directory_list = []

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
            total_directory_list.append(os.path.join(starting_directory, dir))

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

superloop_rounds = 10


# MUST FIGURE OUT HOW TO SAVE THE WEIGHTS IN DIFFERENT PLACES
superloop_maximum_accuracy = 0
superloop_testing_loss_with_maximum_accuracy = 0
superloop_epoch_with_maximum_accuracy = 0
superloop_counter_maximum_accuracy = 0

superloop_minimum_testing_loss = 100000
superloop_accuracy_with_minimum_testing_loss = 0
superloop_epoch_with_minimum_testing_loss = 0
superloop_counter_minimum_testing_loss = 0

for superloop_counter in range(superloop_rounds):
    tfback._get_available_gpus = _get_available_gpus

    base_folder= r'D:\MIT_Tumor_Identifcation_Project_Stuff'
    #input_path = base_folder
    weight_save_folder=r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\1_8_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),0.001_learning,rotate)\Superloop_run_{0}".format(superloop_counter)
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X nostandard groundcells macroV5\step7 Rotated 4D matrices (20-50-50)"
    filepath=os.path.join(weight_save_folder,"1_8_21_Test1_balanced_0.001learning_DNN.{epoch:02d}.hdf5")
    filename=os.path.join(weight_save_folder,'training.log')
    checkDirectory(weight_save_folder)

    epoch_name_part_1 = r"1_8_21_Test1_balanced_0.001learning_DNN."
    epoch_name_part_2 = r".hdf5"

    # Define what the model is here
    shape_aux= (3,20,50,50)
    model = md.new_20X_model_4(input_shape=shape_aux)
    model.summary()


    #number_training_examples = 10728
    number_validation_examples = 100
    number_testing_examples = 100
    nb_epochs = 80
    batch_sz=16
    loss_function='categorical_crossentropy'


    adam=optimizers.Adam(clipvalue=1)
    model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge


    int_eph=0
    csv_logger = CSVLogger(filename)
    lrate = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    callbacks_list = [lrate, checkpoint,csv_logger]

    final_training_data,final_training_labels, final_validation_data, final_validation_labels, final_testing_data, final_testing_labels = get_the_rotated_files()
    '''
    training_counter = 1
    validation_counter = 1
    testing_counter = 1
    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            if training_counter <= number_training_examples:
                training_data.append(os.path.join(starting_directory, dir, "Final_5D_array.npy"))
                training_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))
                training_counter = training_counter + 1
            elif validation_counter <= number_validation_examples:
                validation_data.append(os.path.join(starting_directory, dir, "Final_5D_array.npy"))
                validation_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))
                validation_counter = validation_counter + 1
            elif testing_counter <= number_testing_examples:
                testing_data.append(os.path.join(starting_directory, dir, "Final_5D_array.npy"))
                testing_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))
                testing_counter = testing_counter + 1
            '''
    '''
            for path in Path(os.path.join(starting_directory,dir)).rglob("*.NPY"):
                x = str(path.name)
                if x[0] == "F":
                    if x[1] == "i":
                        if x[2] == "n":
                            # It's a matrix!
                            if training_counter == 3300:
                                training_data.append(os.path.join(starting_directory,dir,path.name))
                                training_counter = training_counter + 1
                            elif validation_counter == 131:
                                validation_data.append(os.path.join(starting_directory, dir, path.name))
                                validation_counter = validation_counter + 1
                            elif testing_counter == 131:
                                testing_data.append(os.path.join(starting_directory, dir, path.name))
                                testing_counter = testing_counter + 1
                if x[0] == "L":
                    if x[1] == "a":
                        if x[2] == "b":
                            # It's a label matrix!
                            training_labels.append(os.path.join(starting_directory,dir,path.name))
                '''





    train_gen = nicholas_generator(final_training_data,final_training_labels,batch_sz)
    valid_gen = nicholas_generator(final_validation_data,final_validation_labels,batch_sz)
    test_gen = nicholas_generator(final_testing_data,final_testing_labels,batch_sz)

    config.gpu_options.allow_growth = True
    model.fit_generator(generator = train_gen, epochs=nb_epochs, validation_data=valid_gen, callbacks=callbacks_list, shuffle="batch", initial_epoch=int_eph, verbose = 2)




    # Function to loop over all of the epoch weight savings and record the maximum accuracy and testing loss,
    # as well as the epochs responsible for the high scores.

    maximum_accuracy = 0
    testing_loss_with_maximum_accuracy = 0
    epoch_with_maximum_accuracy = 0

    minimum_testing_loss = 100000
    accuracy_with_minimum_testing_loss = 0
    epoch_with_minimum_testing_loss = 0
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


        test_acc = model.evaluate_generator(generator = test_gen, verbose=1)
        print('\nTest loss:', test_acc)
        if(correct_counter > maximum_accuracy):
            maximum_accuracy = correct_counter
            testing_loss_with_maximum_accuracy = test_acc
            epoch_with_maximum_accuracy = epoch_counter + 1
        if(minimum_testing_loss > test_acc):
            minimum_testing_loss = test_acc
            accuracy_with_minimum_testing_loss = correct_counter
            epoch_with_minimum_testing_loss = epoch_counter + 1
        cells_that_were_wrong_list = []
        wrong_fibroblasts = 0
        wrong_cancer_cells = 0
        is_correct = False
    model.summary()
    print("The maximum accuracy was " + str(maximum_accuracy)
          + " with testing loss " + str(testing_loss_with_maximum_accuracy)
           + " at epoch " + str(epoch_with_maximum_accuracy))
    print("The minimum testing loss was " + str(minimum_testing_loss)
          + " with accuracy " + str(accuracy_with_minimum_testing_loss)
           + " at epoch " + str(epoch_with_minimum_testing_loss))
    print("Total number of epochs = "+ str(nb_epochs))
    print("Total number of wrong cancer cells = " + str(wrong_cancer_cells))
    print("Total number of wrong fibroblasts = " + str(wrong_fibroblasts))
    #print("List of wrong cells: " + str(cells_that_were_wrong_list))
    if (maximum_accuracy > superloop_maximum_accuracy):
        superloop_maximum_accuracy = maximum_accuracy
        superloop_testing_loss_with_maximum_accuracy = testing_loss_with_maximum_accuracy
        superloop_epoch_with_maximum_accuracy = epoch_with_maximum_accuracy
        superloop_counter_maximum_accuracy = superloop_counter
    if (superloop_minimum_testing_loss > minimum_testing_loss):
        superloop_minimum_testing_loss = minimum_testing_loss
        superloop_accuracy_with_minimum_testing_loss = accuracy_with_minimum_testing_loss
        superloop_epoch_with_minimum_testing_loss = epoch_with_minimum_testing_loss
        superloop_counter_minimum_testing_loss = superloop_counter
    #print("Model running is +" + )
    print("The superloop maximum accuracy was " + str(superloop_maximum_accuracy)
          + " with testing loss " + str(superloop_testing_loss_with_maximum_accuracy)
          + " at epoch " + str(superloop_epoch_with_maximum_accuracy)
          + " at superloop loop counter " + str(superloop_counter_maximum_accuracy))
    print("The superloop minimum testing loss was " + str(superloop_minimum_testing_loss)
          + " with accuracy " + str(superloop_accuracy_with_minimum_testing_loss)
          + " at epoch " + str(superloop_epoch_with_minimum_testing_loss)
          + " at superloop loop counter " + str(superloop_counter_minimum_testing_loss))
    print("Total number of epochs across superloop = " + str(nb_epochs))

