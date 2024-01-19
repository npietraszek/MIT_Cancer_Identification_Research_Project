import re

def read_from_validation_dataset_file(epoch_number, the_filepath):
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

the_epoch_number = 2
filepath = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\2_7_21_new_run_1_(new_model_4,nostandard_harsh_bal,tile_track_1600_seperate)\Superloop_run_2\training.log"
read_from_validation_dataset_file(the_epoch_number, filepath)