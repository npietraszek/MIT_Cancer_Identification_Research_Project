'''
3/23/21
Creates a list of the lowest superloop validation losses for the model for each superloop
by reading the training.log for each superloop folder.
Then prints them out and finds the absolute minimum and prints it for the user.

NEEDS TESTING TO MAKE SURE IT CAN CORRECTLY FIND THE LOWEST VALIDATION LOSS FIGURES
'''


import matplotlib.pyplot as plt
import re
import os
# Import validation data

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


min_valid_loss_dict = {}
epoch_count = 30
validation_data_file_path = r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\only_brightfield\10_2_21_test2_(only_brightfield)\Superloop_run_0"
list_of_directories_to_walk = next(os.walk(validation_data_file_path))[1]

for superloop_directory in list_of_directories_to_walk:
    validation_loss_data = []
    current_minimum_loss = 10000000
    epoch_with_lowest_loss = 0
    file1 = open(os.path.join(validation_data_file_path,superloop_directory, "training.log"), "r")
    datasets = file1.readlines()
    print(len(datasets))
    print("Dataset count above")
    for epoch_number in range(len(datasets) - 1):
        newest_value = read_validation_loss_from_dataset_log(epoch_number, os.path.join(validation_data_file_path,superloop_directory, "training.log"))
        validation_loss_data.append(newest_value)
        if newest_value < current_minimum_loss:
            current_minimum_loss = newest_value
            # Add one here because the text file starts with 0 and our files start with 1
            epoch_with_lowest_loss = epoch_number + 1
    minimum_validation_loss = min(validation_loss_data)
    min_valid_loss_dict[superloop_directory + " epoch " + str(epoch_with_lowest_loss)] = (minimum_validation_loss)

counter = 0
absolute_min_valid_loss = 1000000
absolute_min_superloop_counter = 0
absolute_min_epoch = 0
for epoch_number in min_valid_loss_dict:
    print("Superloop " + str(counter) + " has minimum validation loss " + str(min_valid_loss_dict[epoch_number]) +
          " at epoch " + epoch_number)

    if min_valid_loss_dict[epoch_number] < absolute_min_valid_loss:
        absolute_min_valid_loss = min_valid_loss_dict[epoch_number]
        absolute_min_superloop_counter = counter
        absolute_min_epoch = epoch_number
    counter = counter + 1
print("The absolute minimum validation loss is " + str(absolute_min_valid_loss) + " at superloop counter " +
      str(absolute_min_superloop_counter) + " at epoch " + str(absolute_min_epoch))


