



from keras import optimizers
from gpu_config import _get_available_gpus
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
import keras.backend as tfback
from models import nicholas_models_phase_2 as md
from generators import nicholas_generator
from loss_function_utils import step_decay_creator
import numpy as np
import os

# Fixes relative import issues
import sys
sys.path.append("")
from Common_Utils.machine_learning.feeder_functions import prepare_randomized_cell_datasets
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.machine_learning.post_test_utils import read_validation_loss_from_dataset_log

'''
Main machine learning run program. Runs a "superloop" of multiple machine learning epoch runs with cells randomized into training, validation, and testing datasets.
Each superloop iteration has a different randomization of training dataset and validation dataset and testing dataset.
Each superloop writes its results to a text file in the directory_to_write_to.
Each superloop also writes its datasets into a text file so that each machine learning model can be re-tested whenever necessary.
GPU usage is configured.
Allows parameters to control all aspects of the machine learning and the superloop itself.
Parameters
----------
parent_weight_save_folder : str
    The folder where the weights will be saved. The superloop will create a new folder inside this one for each superloop iteration
    to hold the results of the current superloop.
epoch_name_part_1 : str

'''
def super_train_loop(parent_weight_save_folder, epoch_name_part_1,
                     weight_file_name, model, model_name_to_use, text_results_file_name, directory_to_write_to,
                     superloop_rounds=1, batch_sz=16, nb_epochs=30, 
                     loss_function='categorical_crossentropy', 
                     step_decay_model = step_decay_creator(initial_lrate=0.0001,drop=0.5,epochs_drop=5.0),
                     optimizer = optimizers.Adam(clipvalue=1),
                     print_results = False):

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #from keras import backend as K
    #K.set_image_dim_ordering('tf')
    if print_results == True:
        print("tf.__version__ is", tf.__version__)
        print("tf.keras.__version__ is:", tf.keras.__version__)
    # Define what the model is here
    model.summary()
    step_decay_parameters, step_decay = step_decay_model


    # MUST FIGURE OUT HOW TO SAVE THE WEIGHTS IN DIFFERENT PLACES
    superloop_maximum_accuracy = 0
    superloop_validation_loss_with_maximum_accuracy = 0
    superloop_epoch_with_maximum_validation_accuracy = 0
    superloop_counter_maximum_validation_accuracy = 0

    superloop_minimum_validation_loss = 100000
    superloop_accuracy_with_minimum_validation_loss = 0
    superloop_epoch_with_minimum_validation_loss = 0
    superloop_counter_minimum_validation_loss = 0
    for superloop_counter in range(superloop_rounds):
        tfback._get_available_gpus = _get_available_gpus

        #input_path = base_folder
        weight_save_folder= parent_weight_save_folder + r"\Superloop_run_{0}".format(superloop_counter)
        shortened_weight_save_folder = os.path.split(weight_save_folder)[0]
        filepath=os.path.join(weight_save_folder, weight_file_name)
        filename=os.path.join(weight_save_folder,'training.log')
        checkDirectory(weight_save_folder)


        # number_training_examples = 10728
        # number_validation_examples = 100
        number_testing_examples = 100

        model.compile(loss=loss_function, optimizer=optimizer) #binary_cross_entropy, hinge


        int_eph=0
        csv_logger = CSVLogger(filename)
        lrate = LearningRateScheduler(step_decay)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
        callbacks_list = [lrate, checkpoint,csv_logger]

        final_training_data, final_training_labels, final_validation_data, final_validation_labels, final_testing_data, \
        final_testing_labels, final_saved_testing_data, final_saved_testing_labels = prepare_randomized_cell_datasets()

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

        config.gpu_options.allow_growth = True
        model.fit_generator(generator = train_gen, epochs=nb_epochs, validation_data=valid_gen, callbacks=callbacks_list, shuffle="batch", initial_epoch=int_eph, verbose = 2)

        # Function to loop over all of the epoch weight savings and record the maximum accuracy and testing loss,
        # as well as the epochs responsible for the high scores.
        # Testing the test dataset for each epoch
        if print_results == True:
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
            if print_results == True:
                print("Current epoch is " + str(epoch_counter))
            if epoch_counter + 1 < 10:
                model.load_weights(os.path.join(weight_save_folder,epoch_name_part_1 + "0" + str(epoch_counter + 1) + r".hdf5"))
            else:
                model.load_weights(os.path.join(weight_save_folder, epoch_name_part_1 + str(epoch_counter + 1) + r".hdf5"))
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
        if print_results == True:
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
        if print_results == True:
            print("The superloop maximum accuracy was " + str(superloop_maximum_accuracy)
                + " with validation loss " + str(superloop_validation_loss_with_maximum_accuracy)
                + " at epoch " + str(superloop_epoch_with_maximum_validation_accuracy)
                + " at superloop loop counter " + str(superloop_counter_maximum_validation_accuracy))
            print("The superloop minimum validation loss was " + str(superloop_minimum_validation_loss)
                + " with accuracy " + str(superloop_accuracy_with_minimum_validation_loss)
                + " at epoch " + str(superloop_epoch_with_minimum_validation_loss)
                + " at superloop loop counter " + str(superloop_counter_minimum_validation_loss))
            print("Total number of epochs across superloop = " + str(nb_epochs))

        # NEEDS TESTING
        if epoch_with_minimum_validation_loss < 10:
            model.load_weights(
                os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter)), epoch_name_part_1 + "0" + str(epoch_with_minimum_validation_loss) + r".hdf5"))
        else:
            model.load_weights(
                os.path.join(shortened_weight_save_folder, r"Superloop_run_{0}".format(str(superloop_counter)), epoch_name_part_1 + str(epoch_with_minimum_validation_loss) + r".hdf5"))

        correct_counter = 0
        for i in range(len(final_saved_testing_data)):
            is_correct = False
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
            if print_results == True:
                print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_saved_testing_data[i])
        if print_results == True:
            print("On all the final_saved_testing_data, the machine learning got " + str(correct_counter) + " correct out of " + str(len(final_saved_testing_data))
                + " with accuracy " + str(correct_counter / len(final_saved_testing_data) * 100) + "%")
        #     '''
        #    the_new_pathing = find_original_cell_ROI(final_testing_directories_list[i], device_directory_list[i], r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Reconstruction\new 20x batch Testing with Tuan's V5 Macro")
        #    print("The path found was " + str(the_new_pathing))
        #    csv_getter(the_new_pathing,model.predict(current_example))
        #    '''

        #final_saved_testing_data

    checkDirectory(directory_to_write_to)

    file1 = open(os.path.join(directory_to_write_to,text_results_file_name),"a")#append mode
    file1.write("1_1_22_test_results_29  \n")
    file1.write("Step decay used: initial_lrate =  " + step_decay_parameters[0] + ", drop = " + step_decay_parameters[1] + ", epochs_drop = " + step_decay_parameters[2] + "\n")
    file1.write("Model used was " + model_name_to_use + "\n")
    file1.write("The superloop maximum accuracy was " + str(superloop_maximum_accuracy)
            + " with validation loss " + str(superloop_validation_loss_with_maximum_accuracy)
            + " at epoch " + str(superloop_epoch_with_maximum_validation_accuracy)
            + " at superloop loop counter " + str(superloop_counter_maximum_validation_accuracy) + "\n" +
            ("The superloop minimum validation loss was " + str(superloop_minimum_validation_loss)
            + " with accuracy " + str(superloop_accuracy_with_minimum_validation_loss)
            + " at epoch " + str(superloop_epoch_with_minimum_validation_loss)
            + " at superloop loop counter " + str(superloop_counter_minimum_validation_loss)))
    file1.write("On all the final_saved_testing_data, the machine learning got " + str(correct_counter) + " correct out of " + str(len(final_saved_testing_data))
                + " with accuracy " + str(correct_counter / len(final_saved_testing_data) * 100) + "%")
    file1.write("       ----END---- \n \n \n")
    file1.close()


'''
def old_superloop_function_1_3_21(parent_weight_save_folder, filepath, filename, epoch_name_part_1, epoch_name_part_2, superloop_rounds, model):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # from keras import backend as K
    # K.set_image_dim_ordering('tf')

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
        weight_save_folder = parent_weight_save_folder + r"\Superloop_run_{0}".format(superloop_counter)

        tfback._get_available_gpus = _get_available_gpus


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
        lrate = LearningRateScheduler(step_decay_creator(initial_lrate=0.001, drop=0.5, epochs_drop=5.0))
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
        callbacks_list = [lrate, checkpoint,csv_logger]

        final_training_data,final_training_labels, final_validation_data, final_validation_labels, final_testing_data, final_testing_labels = get_the_rotated_files()

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
'''

