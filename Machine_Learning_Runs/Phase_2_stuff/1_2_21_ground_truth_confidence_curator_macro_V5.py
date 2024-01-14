'''
8/10/20
THIS CODE MUST BE USED TO CURATE ANY IMAGES

Code to find every cell tif (not any of the other processing tifs), and determine if its accuracy is over 70%.
If it is, the code will copy the file and paste it into a new folder
where all the cells can be macroed on.

Glob path: just put in path instead of path.name


'''




import os
import random
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import shutil
import glob
from pathlib import Path
import re

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)




starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step5 4D matrices"
new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step5.5 curated 4D matrices (20-50-50)"
unmatching_cell_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\nonmatching_cells"
checkDirectory(starting_directory)
checkDirectory(new_directory)

fibroblast_counter = 0
cancer_cell_counter = 0
total_cell_count = 0
matching_ground_truth_and_confidence_cells = 0
not_matching = 0
ground_truth_cancer_confidence_fibroblast = 0
ground_truth_fibroblast_confidence_cancer = 0
letters_of_accuracy = ['a','c','c','u','r','a','c','y']
for root, dirs, files in os.walk(starting_directory):
    for the_dir in dirs:
        ground_truth_is_fibroblast = False
        ground_truth_is_cancer = False
        accuracy_test = False
        classification_test = False
        is_cell_fibroblast = False
        is_cell_cancer_cell = False
        x = str(the_dir)
        # Check whether or not the accuracy value is above 90%.
        for ii in range(len(x)):
            if x[ii] == letters_of_accuracy[0]:
                if x[ii + 1] == letters_of_accuracy[1]:
                    if x[ii + 2] == letters_of_accuracy[2]:
                        if x[ii + 3] == letters_of_accuracy[3]:
                            if x[ii + 4] == letters_of_accuracy[4]:
                                if x[ii + 5] == letters_of_accuracy[5]:
                                    if x[ii + 6] == letters_of_accuracy[6]:
                                        if x[ii + 7] == letters_of_accuracy[7]:
                                            total_cell_count = total_cell_count + 1
                                            # letters_before_accuracy = x[:ii+7]
                                            letters_after_accuracy = x[ii + 8:]
                                            print(letters_after_accuracy)
                                            list_of_numbers = (
                                                [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_accuracy)])
                                            print(list_of_numbers)
                                            accuracy_value = list_of_numbers[0]
                                            if accuracy_value >= 0:
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
                    # print("numbers before fb = " + str(list_of_numbers_before_fb))
                    # print("numbers after fb = " + str(list_of_numbers_after_fb))
                    fibroblast_number = list_of_numbers_before_fb[-1]
                    cancer_cell_number = list_of_numbers_after_fb[0]
                    print("fibroblast number = " + str(fibroblast_number))
                    print("cancer cell number = " + str(list_of_numbers_after_fb[0]))
                    if fibroblast_number > cancer_cell_number:
                        if fibroblast_number >= 0.5:
                            classification_test = True
                            is_cell_fibroblast = True
                    else:
                        if cancer_cell_number >= 0.5:
                            classification_test = True
                            is_cell_cancer_cell = True

        # Change this bit to look for the ground truth label file
        path_to_search = os.path.join(starting_directory, the_dir)
        for the_file in Path(path_to_search).rglob('*.npy'):
            x = str(the_file)
            for ii in range(len(x)):
                #print(x[ii:ii+16])
                if x[ii:ii+16] == "Label_matrix.npy":
                    the_label = np.load(str(the_file))
                    if the_label[0] == 1:
                        # ground truth reports fibroblast
                        ground_truth_is_fibroblast = True
                    elif the_label[1] == 1:
                        # ground truth reports cancer
                        ground_truth_is_cancer = True

        if classification_test == True:
            if accuracy_test == True:
                print("You pass! " + str(x) + " passes the test!")
                if ground_truth_is_fibroblast == is_cell_fibroblast or ground_truth_is_cancer == is_cell_cancer_cell:
                    matching_ground_truth_and_confidence_cells = matching_ground_truth_and_confidence_cells + 1

                    the_path = os.path.join(starting_directory,the_dir)
                    destination = new_directory
                    shutil.move(the_path,destination)
                else:
                    not_matching = not_matching + 1
                    if ground_truth_is_fibroblast == True and is_cell_cancer_cell == True:
                        ground_truth_cancer_confidence_fibroblast = ground_truth_cancer_confidence_fibroblast + 1
                    elif ground_truth_is_cancer == True and is_cell_fibroblast == True:
                        ground_truth_fibroblast_confidence_cancer = ground_truth_fibroblast_confidence_cancer + 1

                    the_path = os.path.join(starting_directory,the_dir)
                    destination = unmatching_cell_directory
                    shutil.move(the_path,destination)
                if is_cell_fibroblast:
                    fibroblast_counter = fibroblast_counter + 1
                if is_cell_cancer_cell:
                    cancer_cell_counter = cancer_cell_counter + 1

print("Curation has finished.")
print("Number of cancer cells = " + str(cancer_cell_counter))
print("Number of fibroblast cells = " + str(fibroblast_counter))
print("Total number of cells = " + str(total_cell_count))
print("Total number of cells with not matching ground truth and confidence classifications = " + str(not_matching))
print("Cells with ground truth reporting cancer, and confidence reporting fibroblast: " + str(ground_truth_cancer_confidence_fibroblast))
print("Cells with ground truth reporting fibroblast, and confidence reporting cancer: " + str(ground_truth_fibroblast_confidence_cancer))
print("Total number of cells with matching ground truth and confidence classifcations = " + str(matching_ground_truth_and_confidence_cells))