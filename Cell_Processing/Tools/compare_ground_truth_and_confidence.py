'''
8/10/20
Used to count the number of cells where both our ground truth and confidence classification agree.
Optionally can be used to copy the cells that pass the accuracy test to a new directory.
'''




import os
from pathlib import Path
import re
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.find_accuracy_number import find_accuracy_number
import shutil


def compare_ground_truth_and_confidence(starting_directory, new_directory, test=False):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)

    fibroblast_counter = 0
    cancer_cell_counter = 0
    total_cell_count = 0
    matching_ground_truth_and_confidence_cells = 0
    for the_file in Path(starting_directory).rglob('*.tif'):
        x = str(the_file)

        ground_truth_is_fibroblast = False
        ground_truth_is_cancer = False
        accuracy_test = False
        classification_test = False
        is_cell_fibroblast = False
        is_cell_cancer_cell = False

        # Check whether or not the accuracy value is above 90%.
        accuracy_value = find_accuracy_number(x)
        if accuracy_value >= 50:
            accuracy_test = True

        # Checking the probability values for the cell classifcation.
        for ii in range(len(x)):
            if x[ii] == "F":
                if x[ii + 1] == "b":
                    letters_before_fb = x[:ii]
                    letters_after_fb = x[ii:]
                    if test == True:
                        print("letters before fb = " + str(letters_before_fb))
                        print("letters after fb = " + str(letters_after_fb))
                    list_of_numbers_before_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_before_fb)]
                    list_of_numbers_after_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_fb)]
                    fibroblast_number = list_of_numbers_before_fb[-1]
                    cancer_cell_number = list_of_numbers_after_fb[0]
                    if test == True:
                        print("fibroblast number = " + str(fibroblast_number))
                        print("cancer cell number = " + str(list_of_numbers_after_fb[0]))
                    if fibroblast_number > cancer_cell_number:
                        if fibroblast_number >= 0.9:
                            classification_test = True
                            is_cell_fibroblast = True
                    else:
                        if cancer_cell_number >= 0.9:
                            classification_test = True
                            is_cell_cancer_cell = True
                    directory_to_find_cell_in = os.path.split(x)[0]
                    for the_file in Path(directory_to_find_cell_in).rglob('*.tif'):
                        ground_truth_file_name = str(the_file)
                        index = ground_truth_file_name.find("celltype")
                        if index != -1:
                            # Ground truth file found
                            if ground_truth_file_name[index + 9] == "f":
                                if ground_truth_file_name[index+10] == "i":
                                    # ground truth reports fibroblast
                                    ground_truth_is_fibroblast = True
                            elif ground_truth_file_name[index + 9] == "c":
                                if ground_truth_file_name[index + 10] == "a":
                                    # ground truth reports cancer
                                    ground_truth_is_cancer = True

        if classification_test == True:
            if accuracy_test == True:
                print("You pass! " + str(x) + " passes the test!")
                if ground_truth_is_fibroblast == is_cell_fibroblast or ground_truth_is_cancer == is_cell_cancer_cell:
                    matching_ground_truth_and_confidence_cells = matching_ground_truth_and_confidence_cells + 1
                if is_cell_fibroblast:
                    fibroblast_counter = fibroblast_counter + 1
                if is_cell_cancer_cell:
                    cancer_cell_counter = cancer_cell_counter + 1

    print("Curation has finished.")
    print("Number of cancer cells = " + str(cancer_cell_counter))
    print("Number of fibroblast cells = " + str(fibroblast_counter))
    print("Total number of cells = " + str(total_cell_count))
    print("Total number of cells with matching ground truth and confidence classifcations = " + str(matching_ground_truth_and_confidence_cells))
    return cancer_cell_counter,fibroblast_counter,total_cell_count,matching_ground_truth_and_confidence_cells

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\20X images Decile Checks\0% decile check"
    compare_ground_truth_and_confidence(starting_directory,new_directory)