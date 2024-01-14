'''
8/29/20

Program to remove excess cancer cells by removing the ones with the lowest confidences in classifcation.



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

def remove_brightfield_and_reflective_from_4D(array_to_remove_from):
    new_array = np.zeros((1,20,50,50))
    for b in range(len(array_to_remove_from[0])):
        for c in range(len(array_to_remove_from[0][0])):
            for d in range(len(array_to_remove_from[0][0][0])):
                new_array[0][b][c][d] = array_to_remove_from[0][b][c][d]
    return new_array

def stack_DAPI_into_5D(array_to_move, array_to_stack_onto):
    for z in range(len(array_to_move)):
        for y in range(len(array_to_move[0])):
            for x in range(len(array_to_move[0][0])):
                array_to_stack_onto[0][z][y][x] = array_to_move[z][y][x]
    return array_to_stack_onto

def stack_Reflective_into_5D(array_to_move, array_to_stack_onto):
    for z in range(len(array_to_move)):
        for y in range(len(array_to_move[0])):
            for x in range(len(array_to_move[0][0])):
                array_to_stack_onto[1][z][y][x] = array_to_move[z][y][x]
    return array_to_stack_onto

def stack_Transmission_brightfield_into_5D(array_to_move, array_to_stack_onto):
    for z in range(len(array_to_move)):
        for y in range(len(array_to_move[0])):
            for x in range(len(array_to_move[0][0])):
                array_to_stack_onto[2][z][y][x] = array_to_move[z][y][x]
    return array_to_stack_onto

# Pads the matrices in the list based on the maximum length and width of the matrices in the list.
def pad_matrices_in_list(the_list):
    max_y = 0
    max_x = 0
    for z in range(len(the_list)):
        if len(the_list[z]) > max_y:
            max_y = len(the_list[z])
        for y in range(len(the_list[z])):
            if len(the_list[z][y]) > max_x:
                max_x = len(the_list[z][y])
    padding_matrix = np.zeros((len(the_list),max_y,max_x))
    for z in range(len(the_list)):
        for y in range(len(the_list[z])):
            for x in range(len(the_list[z][y])):
                padding_matrix[z][y][x] = the_list[z][y][x]
    return padding_matrix

# Pads the matrices in the list based on a standard size.
# (SHOULD NOT BE USED IF THE STANDARD SIZE IS SMALLER THAN ANY OF THE MATRICES IN IT)
def pad_matrices_in_list_to_standard(the_list, z_length,y_length,x_length):
    padding_matrix = np.zeros((z_length,y_length,x_length))
    for z in range(len(the_list)):
        for y in range(len(the_list[z])):
            for x in range(len(the_list[z][y])):
                padding_matrix[z][y][x] = the_list[z][y][x]
    return padding_matrix



starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X nostandard groundcells macroV5\step5.5 curated 4D matrices (20-50-50)"
new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X nostandard groundcells macroV5\step6 Balanced 4D matrices (20-50-50)"
moving_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X nostandard groundcells macroV5\unused_cancer_cells"
checkDirectory(starting_directory)
checkDirectory(new_directory)
checkDirectory(moving_directory)
shape_of_3D_matrices = [20,50,50]

# We need to remove cancer cells
cancer_cell_counter = 0
fibroblast_counter = 0
letters_of_accuracy = ['a','c','c','u','r','a','c','y']
list_of_cancer_cells = []
list_of_accuracy_values = []
list_of_confidences = []

list_of_worst_cancer_cells = []


for root, dirs, files in os.walk(starting_directory):
    for dir in dirs:
        x = str(dir)

        accuracy_test = False
        classification_test = False
        is_cell_fibroblast = False
        is_cell_cancer_cell = False


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
                            print("You pass! " + str(x) + " passes the test!")
                            the_path = os.path.join(starting_directory, dir)
                            destination = os.path.join(new_directory, dir)
                            shutil.copytree(the_path, destination)
                            fibroblast_counter = fibroblast_counter + 1
                    else:
                        if cancer_cell_number >= 0.5:
                            classification_test = True
                            is_cell_cancer_cell = True
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
                                                                # letters_before_accuracy = x[:ii+7]
                                                                letters_after_accuracy = x[ii + 8:]
                                                                print(letters_after_accuracy)
                                                                list_of_numbers = (
                                                                    [float(s) for s in re.findall(r'-?\d+\.?\d*',
                                                                                                  letters_after_accuracy)])
                                                                print(list_of_numbers)
                                                                accuracy_value = list_of_numbers[0]
                                                                list_of_cancer_cells.append(dir)
                                                                list_of_confidences.append(cancer_cell_number)
                                                                list_of_accuracy_values.append(accuracy_value)
                                                                cancer_cell_counter = cancer_cell_counter + 1

for x in range(1879):
    current_minimum = min(list_of_confidences)
    for i in range(len(list_of_cancer_cells)):
        if list_of_confidences[i] == current_minimum:
            # A minimum confidence has been found. Pop the values there...
            list_of_accuracy_values.pop(i)

            # Move this cancer cell elsewhere...
            print(list_of_cancer_cells[i])
            the_path = os.path.join(starting_directory, list_of_cancer_cells[i])
            destination = os.path.join(moving_directory, list_of_cancer_cells[i])
            shutil.copytree(the_path, destination)
            list_of_cancer_cells.pop(i)
            list_of_confidences.pop(i)
            cancer_cell_counter = cancer_cell_counter - 1
            break
print("Number of cancer cells = " + str(cancer_cell_counter))
print("Number of fibroblast cells = " + str(fibroblast_counter))
# After the weakest cancer cells have been removed, continue
for x in list_of_cancer_cells:
    print("You pass! " + str(x) + " passes the test!")
    the_path = os.path.join(starting_directory, x)
    destination = os.path.join(new_directory, x)
    shutil.copytree(the_path, destination)

print("Curation has finished.")
print("Number of cancer cells = " + str(cancer_cell_counter))
print("Number of fibroblast cells = " + str(fibroblast_counter))