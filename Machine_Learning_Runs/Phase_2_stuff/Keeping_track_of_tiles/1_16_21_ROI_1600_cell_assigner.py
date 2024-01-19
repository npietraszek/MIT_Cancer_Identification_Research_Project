'''
7/2/20
Assigns 1600 cells to the saved_1600_cell_directory and the rest to the new_directory.
'''




import os
import random
import shutil
from pathlib import Path
import random
from Common_Utils.checkDirectory import checkDirectory


'''
Assigns 1600 cells to the saved_1600_cell_directory and the rest to the new_directory.
Used to get a random testing dataset for the machine learning to work off of.
'''
def ROI_1600_cell_assigner(starting_directory, saved_1600_cell_directory, new_directory):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)

    # Build a dictionary of ROI folders and the numbers of cells they contain from the .txt files.
    dictionary_of_file_counters = {}
    for the_file in Path(starting_directory).rglob('*.txt'):
        path_to_folder = os.path.split(the_file)[0]
        file1 = open(the_file,"r+")
        number_cells = file1.read()
        dictionary_of_file_counters[str(path_to_folder)] = number_cells

    # Take ROI folders from the dictionary of cells until 1600 cells are obtained in total from the dictionary
    # We do 400 because each file has 4 rotated images of cells
    total_number = 0
    while total_number < 400:
        cell_directory, directory_cell_number = random.choice(list(dictionary_of_file_counters.items()))
        del dictionary_of_file_counters[cell_directory]
        print("Taking directory " + str(cell_directory) + " as part of the saved 1600.")
        print("Total number of cells within: " + str(directory_cell_number))
        the_name_of_directory = os.path.split(cell_directory)[1]
        shutil.copytree(cell_directory, os.path.join(saved_1600_cell_directory, the_name_of_directory))
        total_number = total_number + int(directory_cell_number)

        print("Current total number: " + str(total_number))


    # Take the rest of the cells and put them into a different folder for machine learning to draw from
    # for all of training, validation, and testing
    for cell_key in dictionary_of_file_counters:
        the_name_of_cell_directory = os.path.split(cell_key)[1]
        print("Moving directory " + str(cell_key) + " over to the normal machine learning folder")
        shutil.copytree(cell_key,os.path.join(new_directory, the_name_of_cell_directory))

        # some house keeping to make sure we got all the cells
        total_number = total_number +  int(dictionary_of_file_counters[cell_key])

    print("Total number of cells moved to both directories: " + str(total_number))


if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step8 Rotated ROI sorted 4D"
    saved_1600_cell_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_1600_cells"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_without_1600"
    ROI_1600_cell_assigner(starting_directory, saved_1600_cell_directory, new_directory)