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



for the_loop_counter in range(0,10):
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\20X images Decile Checks\{0}".format(the_loop_counter) + r"0% decile check"
    directory_to_write_to = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\20X images Decile Checks"
    checkDirectory(starting_directory)
    checkDirectory(new_directory)

    fibroblast_counter = 0
    cancer_cell_counter = 0
    total_cell_count = 0
    letters_of_accuracy = ['a','c','c','u','r','a','c','y']
    for the_file in Path(starting_directory).rglob('*.tif'):
        x = str(the_file)

        accuracy_test = False
        classification_test = False
        is_cell_fibroblast = False
        is_cell_cancer_cell = False

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
                                            float(10*the_loop_counter)
                                            if accuracy_value > float(10*the_loop_counter) and accuracy_value <= float(10*(the_loop_counter+1)):
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
        if classification_test == True:
            if accuracy_test == True:
                print("You pass! " + str(x) + " passes the test!")
                the_path = os.path.join(starting_directory,the_file)
                destination = new_directory
                #shutil.copy(the_path,destination)
                if is_cell_fibroblast:
                    fibroblast_counter = fibroblast_counter + 1
                if is_cell_cancer_cell:
                    cancer_cell_counter = cancer_cell_counter + 1

    print("Curation has finished.")
    print("Number of cancer cells = " + str(cancer_cell_counter))
    print("Number of fibroblast cells = " + str(fibroblast_counter))
    print("Total number of cells = " + str(total_cell_count))
    file1 = open(os.path.join(directory_to_write_to, r"10_29_20_test_results_{0}".format(the_loop_counter) + "0% - {0}0%".format(the_loop_counter + 1)  + "_decile.txt"), "a")  # append mode
    file1.write("10_29_20_test_results_{0}".format(the_loop_counter) + "0% - {0}0%".format(the_loop_counter + 1)  + "_decile" + "\n")
    file1.write("Number of cancer cells = " + str(cancer_cell_counter) + "\n")
    file1.write("Number of fibroblast cells = " + str(fibroblast_counter) + "\n")
    file1.write("Number of cells under current decile = " + str(cancer_cell_counter+fibroblast_counter) + "\n")
    file1.write("Total number of cells = " + str(total_cell_count) + "\n")
    file1.write("       ----END---- \n \n \n")
    file1.close()