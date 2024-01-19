'''
8/10/20
THIS CODE MUST BE USED TO CURATE ANY IMAGES

Code to find every cell tif (not any of the other processing tifs), and determine if its accuracy is over 70%.
If it is, the code will copy the file and paste it into a new folder
where all the cells can be macroed on.

Glob path: just put in path instead of path.name


'''
import os
import shutil
from pathlib import Path
import re
from Common_Utils.checkDirectory import checkDirectory

'''
Searches out cells that have an accuracy value above the accuracy_standard and a classification value above the classification_standard inside the starting_directory.
Then copies the cells into the target_directory.

Parameters
----------
starting_directory : string
    The directory to start searching in recursively.
target_directory : string
    The directory to copy the cells into.
accuracy_standard : float
    The minimum accuracy value that the cell must have to be copied.
classification_standard: float
    The minimum confidence threshold a cell has to have for us to be confident it is a cancer cell or fibroblast,
    and therefore worth copying.

Returns
-------
cancer_cell_counter : int
    The number of cancer cells that were copied.
fibroblast_counter : int
    The number of fibroblast cells that were copied.
total_cell_count : int
    The total number of cells that were copied.

'''
def cell_searcher(starting_directory, target_directory, accuracy_standard = 0, classification_standard = 0.5):
    checkDirectory(starting_directory)
    checkDirectory(target_directory)
    fibroblast_counter = 0
    cancer_cell_counter = 0
    total_cell_count = 0
    for the_file in Path(starting_directory).rglob('*.tif'):
        x = str(the_file)

        accuracy_test = False
        classification_test = False
        is_cell_fibroblast = False
        is_cell_cancer_cell = False

        # Check whether or not the accuracy value is above 90%.
        accuracy_index = x.find("accuracy")
        if accuracy_index != -1:
            letters_after_accuracy = x[accuracy_index+8:]
            list_of_numbers = ([float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_accuracy)])
            accuracy_value = list_of_numbers[0]
            if accuracy_value >= accuracy_standard:
                accuracy_test = True
            # Checking the probability values for the cell classifcation.
            for ii in range(len(x)):
                if x[ii] == "F":
                    if x[ii + 1] == "b":
                        letters_before_fb = x[:ii]
                        letters_after_fb = x[ii:]
                        list_of_numbers_before_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_before_fb)]
                        list_of_numbers_after_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_fb)]
                        fibroblast_number = list_of_numbers_before_fb[-1]
                        cancer_cell_number = list_of_numbers_after_fb[0]
                        if fibroblast_number > cancer_cell_number:
                            if fibroblast_number >= classification_standard:
                                classification_test = True
                                is_cell_fibroblast = True
                        else:
                            if cancer_cell_number >= classification_standard:
                                classification_test = True
                                is_cell_cancer_cell = True
        if classification_test == True:
            if accuracy_test == True:
                # The cells pass both the classification and accuracy tests.

                the_path = os.path.join(starting_directory,the_file)
                destination = target_directory
                shutil.copy(the_path,destination)

                # renaming the file to include the tile it is from.
                if "chip 1 and 2" in str(the_file):
                    # chip 1 and 2 of device 1 needed
                    initial_file_name = os.path.split(the_file)[1]
                    new_file_name = r"device 1 chip 1 and 2 " + str(initial_file_name)
                elif "device 3 chip 3" in str(the_file):
                    # device 3 chip 3
                    initial_file_name = os.path.split(the_file)[1]
                    new_file_name = r"device 3 chip 3 " + str(initial_file_name)
                elif "chip 3" in str(the_file):
                    # chip 3 of device 1 needed
                    initial_file_name = os.path.split(the_file)[1]
                    new_file_name = r"device 1 chip 3 " + str(initial_file_name)
                elif "device 2" in str(the_file):
                    # device 2 needed
                    initial_file_name = os.path.split(the_file)[1]
                    new_file_name = r"device 2 " + str(initial_file_name)
                elif "device 3 chip 1 2 3" in str(the_file):
                    # device 3 chip 1 2 3 needed
                    initial_file_name = os.path.split(the_file)[1]
                    new_file_name = r"device 3 chip 1 2 3 " + str(initial_file_name)

                else:
                    raise ValueError("No location was detected for where this cell came from...")

                rename_location = os.path.join(destination,new_file_name)
                copied_file = os.path.join(destination, initial_file_name)
                shutil.move(copied_file,rename_location)
                if is_cell_fibroblast:
                    fibroblast_counter = fibroblast_counter + 1
                if is_cell_cancer_cell:
                    cancer_cell_counter = cancer_cell_counter + 1

    print("Step 1 Cell Searching Curation has finished.")
    print("Number of cancer cells = " + str(cancer_cell_counter))
    print("Number of fibroblast cells = " + str(fibroblast_counter))
    print("Total number of cells = " + str(total_cell_count))
    return (cancer_cell_counter, fibroblast_counter, total_cell_count)



if __name__ == "__main__":
    starting_directory = r"d:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\BatchTestingFolder"
    target_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\CodeTestFolder2\step1 TIFFs and PNGs"
    cell_searcher(starting_directory, target_directory, accuracy_standard = 0, classification_standard = 0.5)




