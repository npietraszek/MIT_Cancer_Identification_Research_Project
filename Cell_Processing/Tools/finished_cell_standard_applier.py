'''
8/10/20
THIS CODE MUST BE USED TO CURATE ANY IMAGES

Code to find every cell tif (not any of the other processing tifs), and determine if its accuracy is over a certain threshold.
If it is, the code will copy the file and paste it into a new folder
where all the cells can be macroed on.

Glob path: just put in path instead of path.name


'''




import os
import shutil
import re
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.find_accuracy_number import find_accuracy_number


def finished_cell_standard_applier(starting_directory, new_directory, test=False):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)

    fibroblast_counter = 0
    cancer_cell_counter = 0
    total_cell_count = 0
    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            x = str(dir)

            accuracy_test = False
            classification_test = False
            is_cell_fibroblast = False
            is_cell_cancer_cell = False
            accuracy_value = find_accuracy_number(x)
            if accuracy_value >= 50:
                accuracy_test = True
            # Checking the probability values for the cell classifcation.
            for ii in range(len(x)):
                if x[ii] == "F":
                    if x[ii + 1] == "b":
                        letters_before_fb = x[:ii]
                        letters_after_fb = x[ii:]
                        
                        print("letters before fb = " + str(letters_before_fb))
                        print("letters after fb = " + str(letters_after_fb))
                        list_of_numbers_before_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_before_fb)]
                        list_of_numbers_after_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_fb)]
                        fibroblast_number = list_of_numbers_before_fb[-1]
                        cancer_cell_number = list_of_numbers_after_fb[0]
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
            if classification_test == True:
                if accuracy_test == True:
                    print("You pass! " + str(x) + " passes the test!")
                    the_path = os.path.join(starting_directory,dir)
                    destination = os.path.join(new_directory,dir)
                    shutil.copytree(the_path,destination)
                    if is_cell_fibroblast:
                        fibroblast_counter = fibroblast_counter + 1
                    if is_cell_cancer_cell:
                        cancer_cell_counter = cancer_cell_counter + 1

    print("Curation has finished.")
    print("Number of cancer cells = " + str(cancer_cell_counter))
    print("Number of fibroblast cells = " + str(fibroblast_counter))
    print("Total number of cells = " + str(total_cell_count))


if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X (no standard) Images Macro V5\step7 Rotated 4D matrices (20-50-50)"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X (90 confidence standard) Macro V5\step7 Rotated 4D matrices"
    finished_cell_standard_applier(starting_directory,new_directory)