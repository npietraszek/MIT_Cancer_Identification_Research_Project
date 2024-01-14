'''
3/23/21
Creates a list of the highest superloop accuracies for the model for each superloop
by testing on the test dataset for each model for each epoch.
Then prints them out and finds the absolute maximum accuracy and prints it for the user.

NEEDS TESTING TO MAKE SURE IT CAN CORRECTLY FIND THE HIGHEST ACCURACIES...
'''


import matplotlib.pyplot as plt
import re
import os
from MIT_Tumor_Identifcation_Project.Machine_learning_runs.Phase_2_stuff import nicholas_models_phase_2_new_testing as md
import numpy as np
# Import testing data

def split_strip_list_conversion(initial_text):
    initial_text = initial_text.replace("'", "")
    initial_text = initial_text.replace("\\\\", "\\")
    initial_text.replace("]", "")
    initial_text.replace("\n", "")
    initial_text.replace("[", "")
    res = initial_text.strip('][\n').split(', ')
    return res

def test_the_model(epoch_number, dataset_filepath):
    shape_aux = (3, 20, 50, 50)
    model = md.new_20X_model_4(input_shape=shape_aux)
    #list_of_accuracy_values = []
    #list_of_testing_loss = []

    # Must read datasets from text file
    file1 = open(os.path.join(dataset_filepath, r"Datasets.txt"), "r")
    datasets = file1.readlines()

    if epoch_number + 1 < 10:
        path_to_model = os.path.join(dataset_filepath,
                                     epoch_name_part_1 + "0" + str(epoch_number) + epoch_name_part_2)
        model.load_weights(
            path_to_model)
    else:
        path_to_model = os.path.join(dataset_filepath,
                                     epoch_name_part_1 + str(epoch_number) + epoch_name_part_2)
        model.load_weights(path_to_model)
    correct_counter = 0

    final_testing_data = split_strip_list_conversion(datasets[5])
    final_testing_labels = split_strip_list_conversion(datasets[6])
    #final_saved_testing_data = split_strip_list_conversion(datasets[7])
    #final_saved_testing_labels = split_strip_list_conversion(datasets[8])

    #compiled_testing_data = final_testing_data + final_saved_testing_data
    #compiled_testing_labels = final_testing_labels + final_saved_testing_labels
    correct_counter = 0
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

        # print(model.predict(current_example), testing_answer, is_correct, correct_counter, compiled_testing_data[i])
    return correct_counter

epoch_name_part_1 = r"1_16_21_Test1_balanced_0.001learning_DNN."
epoch_name_part_2 = r".hdf5"
max_testing_accuracy_dict = {}
epoch_count = 40
testing_epoch_file_path = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\3_21_21_new_run_2_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)"
list_of_directories_to_walk = next(os.walk(testing_epoch_file_path))[1]

for superloop_directory in list_of_directories_to_walk:
    testing_accuracy_data = []
    current_maximum_accuracy = 0
    epoch_with_highest_accuracy = 0
    for epoch_number in range(epoch_count):
        newest_value = test_the_model(epoch_number, os.path.join(testing_epoch_file_path,superloop_directory))
        testing_accuracy_data.append(newest_value)
        if newest_value > current_maximum_accuracy:
            current_maximum_accuracy = newest_value
            epoch_with_highest_accuracy = epoch_number
    maximum_accuracy = max(testing_accuracy_data)
    max_testing_accuracy_dict[superloop_directory + " epoch " + str(epoch_with_highest_accuracy)] = maximum_accuracy

counter = 0
absolute_max_testing_accuracy = 1000000
absolute_max_superloop_counter = 0
absolute_max_epoch = 0
for epoch_number in max_testing_accuracy_dict:
    print("Superloop " + str(counter) + " has minimum validation loss " + str(max_testing_accuracy_dict[epoch_number]) +
          " at epoch " + epoch_number)

    if max_testing_accuracy_dict[epoch_number] > absolute_max_testing_accuracy:
        absolute_max_testing_accuracy = max_testing_accuracy_dict[epoch_number]
        absolute_max_superloop_counter = counter
        absolute_max_epoch = epoch_number
    counter = counter + 1
print("The absolute minimum validation loss is " + str(absolute_max_testing_accuracy) + " at superloop counter " +
      str(absolute_max_superloop_counter) + " at epoch " + str(absolute_max_epoch))


