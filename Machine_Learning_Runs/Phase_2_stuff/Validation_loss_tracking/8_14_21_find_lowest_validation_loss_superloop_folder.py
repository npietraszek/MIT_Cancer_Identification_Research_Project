'''
8/14/21
Gets the lowest validation loss from the number of epochs and path specified
and returns it along with the epoch number that brought that lowest validation loss.

'''


import matplotlib.pyplot as plt
import re
# Import validation data

def read_validation_loss_from_dataset_log(epoch_number, the_filepath):
    #print("Reading validation from " + str(epoch_number))
    file1 = open((the_filepath), "r")
    datasets = file1.readlines()
    the_line = datasets[epoch_number+1]
    #print("The line is " + str(the_line))
    split_line = re.split("(,)", the_line)
    #print("The split line is " + str(split_line))
    epoch_validation_loss = float(split_line[6][0:-1])
    #print(epoch_validation_loss)
    return epoch_validation_loss

validation_loss_data = []

epoch_count = 30
validation_data_file_path = r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\Half_Training_All_Images\11_19_21_Test19_(half_training)\Superloop_run_0\training.log"
print(validation_data_file_path)

minimum_validation_loss = 99999
min_validation_epoch_number = 0
for epoch_number in range(epoch_count):
    current_validation_loss = read_validation_loss_from_dataset_log(epoch_number, validation_data_file_path)
    validation_loss_data.append(current_validation_loss)
    if current_validation_loss < minimum_validation_loss:
        minimum_validation_loss = current_validation_loss
        min_validation_epoch_number = epoch_number

print("From loop, minimum validation loss is " + str(minimum_validation_loss))
print("From loop, minimum validation loss epoch number is " + str(min_validation_epoch_number + 1))