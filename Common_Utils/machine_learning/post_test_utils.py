import re

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
