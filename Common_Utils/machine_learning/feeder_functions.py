from random import shuffle
import os
import numpy as np

'''
Shuffles the matrices in the list without changing the order of the labels in the labels list.
Returns the shuffled matrices alongside their rotations as lists that the machine learning algorithm can use.
Parameters
----------
labels_list : list
    The list of labels to shuffle.
the_dictionary : dictionary
    The dictionary of matrices to shuffle.
Returns
-------
data_shuf : list
    The shuffled matrices.
labels_shuf : list
    The shuffled labels.
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
    return data_shuf, labels_shuf
'''
Randomly assigns each of of the cells in the starting directory to either training data, validation data, or testing data.
Then, randomly shuffles the 4D matrices in each of the training, validation, and testing data, 
as well as the saved_1600 cells directory, keeping the order of the label matrices intact.               
Returns the shuffled cells alongside their rotations as lists that the machine learning algorithm can use.

Parameters:
----------  
starting_directory : str
    Directory where all the cells except the 1600 cells are saved.     
saved_1600_cells_directory : str
    Directory where the 1600 cells are saved.
test: bool
    Whether or not to print out some test values.
Returns:
shuffled_training_data,
shuffled_training_labels,
shuffled_validation_data,
shuffled_validation_labels, 
shuffled_testing_data, 
shuffled_testing_labels, 
shuffled_saved_testing_data, 
shuffled_saved_testing_labels
'''
def prepare_randomized_cell_datasets(saved_1600_cells_directory, starting_directory, test = False, rotated = True):
    if rotated == True:
        orientations = 4
    else:
        orientations = 1
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
    for x in range(1, orientations+1):
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
            for x in range(1, orientations+1):
                saved_testing_data_dict["saved_testing_data_{0}".format(x)].append(
                    os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                # print(validation_data_dict["validation_data_{0}".format(x)][0])
                # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
            saved_testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

            testing_saving_cancer_counter = testing_saving_cancer_counter + 1
            testing_saving_counter = testing_saving_counter + 1

        else:
            # is_cancer_cell = False
            for x in range(1, orientations+1):
                saved_testing_data_dict["saved_testing_data_{0}".format(x)].append(
                    os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                # print(validation_data_dict["validation_data_{0}".format(x)][0])
                # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
            saved_testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

            testing_saving_fibroblast_counter = testing_saving_fibroblast_counter + 1
            testing_saving_counter = testing_saving_counter + 1
    if test == True:
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
    if test == True:
        print(len(total_directory_list))
    shuffle(total_directory_list)
    for the_dir in total_directory_list:
        the_label_matrix = np.load(os.path.join(the_dir, "Label_matrix.npy"))

        if the_label_matrix[0] == 0:
            # We are dealing with a cancer cell
            # If there aren't enough cancer cells for validation, add this cell to it.
            if validation_cancer_cell_counter <= number_validation_cancer_cells:
                for x in range(1, orientations+1):
                    validation_data_dict["validation_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    #print(validation_data_dict["validation_data_{0}".format(x)][0])
                    #the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
                validation_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                validation_cancer_cell_counter = validation_cancer_cell_counter + 1
                validation_counter = validation_counter + 1
            elif testing_cancer_cell_counter <= number_testing_cancer_cells:
                for x in range(1, orientations+1):
                    testing_data_dict["testing_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                testing_cancer_cell_counter = testing_cancer_cell_counter + 1
                testing_counter = testing_counter + 1
            #elif training_cancer_cell_counter <= number_training_cancer_cells:
            else:
                for x in range(1, orientations+1):
                    training_data_dict["training_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                training_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                training_cancer_cell_counter = training_cancer_cell_counter + 1
                training_counter = training_counter + 1
        else:
            # is_cancer_cell = False
            # If there aren't enough fibroblasts for validation, add this cell to it.
            if validation_fibroblast_counter <= number_validation_fibroblasts:
                for x in range(1, orientations+1):
                    validation_data_dict["validation_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                    # print(validation_data_dict["validation_data_{0}".format(x)][0])
                    # the_matrix = np.load(validation_data_dict["validation_data_{0}".format(x)][0])
                validation_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                validation_fibroblast_counter = validation_fibroblast_counter + 1
                validation_counter = validation_counter + 1

            elif testing_fibroblast_counter <= number_testing_fibroblasts:
                for x in range(1, orientations+1):
                    testing_data_dict["testing_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                testing_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                testing_fibroblast_counter = testing_fibroblast_counter + 1
                testing_counter = testing_counter + 1

            #elif training_fibroblast_counter <= number_training_fibroblasts:
            else:
                for x in range(1, orientations+1):
                    training_data_dict["training_data_{0}".format(x)].append(
                        os.path.join(the_dir, "Final_5D_array_{0}".format(x) + ".npy"))
                training_labels.append(os.path.join(the_dir, "Label_matrix.npy"))

                training_fibroblast_counter = training_fibroblast_counter + 1
                training_counter = training_counter + 1
    if test == True:
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


    assert len(shuffled_training_data) == 2088*orientations
    assert len(shuffled_validation_data) == 100*orientations
    assert len(shuffled_testing_data) == 100*orientations
    assert len(shuffled_saved_testing_data) == 408*orientations

    return shuffled_training_data,shuffled_training_labels,shuffled_validation_data,shuffled_validation_labels, shuffled_testing_data, shuffled_testing_labels, shuffled_saved_testing_data, shuffled_saved_testing_labels