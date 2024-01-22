import re

'''
Searches through the dataset log file and returns the validation loss for the epoch number specified
Needed to determine what was the lowest validation loss epoch for the machine learning run.
Parameters
----------
epoch_number: int
    The epoch number to get the validation loss for
the_filepath: string
    The path to the dataset log file
Returns
-------
epoch_validation_loss: float
    The validation loss for the epoch number specified
'''
def read_validation_loss_from_dataset_log(epoch_number, the_filepath, test=False):
    if test == True:
        print("Reading validation from " + str(epoch_number))
    file1 = open((the_filepath), "r")
    datasets = file1.readlines()
    the_line = datasets[epoch_number+1]
    if test == True:
        print("The line is " + str(the_line))
    split_line = re.split("(,)", the_line)
    if test == True:
        print("The split line is " + str(split_line))
    epoch_validation_loss = float(split_line[6][0:-1])
    if test == True:
        print(epoch_validation_loss)
    return epoch_validation_loss
