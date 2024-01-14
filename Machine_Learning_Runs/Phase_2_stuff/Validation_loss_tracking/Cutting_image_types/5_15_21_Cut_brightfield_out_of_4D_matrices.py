'''
7/7/20

Program to take 4D matrices already generated and cut the brightfield image type out of them, then save them to
a seperate folder with the correct directory name...

Will specialize this for the xy rotated matrices only since those are the ones that count...

Debugging:
Correctly loads the correct number of directories from the right place: True
Correctly loads the matrices: True
Crops the correct data slice that we wanted to crop away (the brightfield): Seems to be True
Correctly saves the data matrices: True
Correctly saves the label matrices: True

Program works as intended!
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

def remove_brightfield_from_4D(array_to_remove_from):
    new_array = np.zeros((2,20,50,50))
    for b in range(len(array_to_remove_from[0])):
        for c in range(len(array_to_remove_from[0][0])):
            for d in range(len(array_to_remove_from[0][0][0])):
                new_array[0][b][c][d] = array_to_remove_from[0][b][c][d]
                new_array[1][b][c][d] = array_to_remove_from[1][b][c][d]
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



# starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Usable_5D_matrices\7_2_20_Rotated_4D_matrices_(harsh_no_strange_cells)"
# new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Usable_5D_matrices\7_7_20_No_Brightfield_rotated_4D_(harsh_no_strange_cells)"
starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\step9 Rotated_ROI_without_1600"
new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\No_brightfield_without_1600"
checkDirectory(starting_directory)
checkDirectory(new_directory)

# shape_of_3D_matrices = [20,30,30]

# D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\step9 Rotated_ROI_1600_cells\device 1 chip 1 and 2 ROI1_02.oib\device 1 chip 1 and 2 ROI1_02.oib - Series 1-1 0000_cell1_0.7695Fb0.2305Tc_accuracy75.5408index6
# D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\step9 Rotated_ROI_1600_cells\device 1 chip 1 and 2 ROI1_02.oib\device 1 chip 1 and 2 ROI1_02.oib - Series 1-1 0000_cell1_0.7695Fb0.2305Tc_accuracy75.5408index6
list_of_directories_to_walk = next(os.walk(starting_directory))[1]
for directory in list_of_directories_to_walk:
    num_cells_file = open(os.path.join(starting_directory, directory,  "Number_of_cells_in_ROIOI.txt"))
    num_cells = num_cells_file.read()
    num_cells_file.close()
    checkDirectory(os.path.join(new_directory, directory))
    file1 = open(os.path.join(new_directory, directory, r"Number_of_cells_in_ROIOI.txt"), "a")  # append mode
    file1.write(num_cells)
    file1.close()
    print("Number of cells file written")


    for root, dirs, files in os.walk(os.path.join(starting_directory,directory)):
        for dir in dirs:
            # Begin searching each directory for the 2D PNGs
            print("Now searching directory " + str(os.path.join(starting_directory,directory,dir)))
            final_array_1 = np.load(os.path.join(starting_directory, directory, dir, "Final_5D_array_1.npy"))
            final_array_2 = np.load(os.path.join(starting_directory, directory, dir, "Final_5D_array_2.npy"))
            final_array_3 = np.load(os.path.join(starting_directory, directory, dir, "Final_5D_array_3.npy"))
            final_array_4 = np.load(os.path.join(starting_directory, directory, dir,"Final_5D_array_4.npy"))
            label_array = np.load(os.path.join(starting_directory, directory, dir, "Label_matrix.npy"))


            new_array_1 = remove_brightfield_from_4D(final_array_1)
            new_array_2 = remove_brightfield_from_4D(final_array_2)
            new_array_3 = remove_brightfield_from_4D(final_array_3)
            new_array_4 = remove_brightfield_from_4D(final_array_4)

            print(new_array_1)

            final_path = os.path.join(new_directory, directory, dir)
            checkDirectory(final_path)
            np.save(os.path.join(final_path, r"Final_5D_array_1.npy"), new_array_1)
            np.save(os.path.join(final_path, r"Final_5D_array_2.npy"), new_array_2)
            np.save(os.path.join(final_path, r"Final_5D_array_3.npy"), new_array_3)
            np.save(os.path.join(final_path, r"Final_5D_array_4.npy"), new_array_4)
            np.save(os.path.join(final_path, r"Label_matrix.npy"), label_array)
            '''
            final_path = os.path.join(new_directory, dir)
    
            checkDirectory(final_path)
    
            np.save(os.path.join(final_path, r"Final_5D_array"), The_final_5D_array)
            np.save(os.path.join(final_path,r"Label_matrix"),The_label_matrix)
    
            # reset the matrices after saving them.
    
            # The 3 3D arrays off this data
            DAPI_3D_array = []
            Reflection_3D_array = []
            Transmission_brightfield_3D_array = []
    
            # The arrays to be finally saved
            The_final_5D_array = np.zeros((3, shape_of_3D_matrices[0], shape_of_3D_matrices[1], shape_of_3D_matrices[2]))
            The_label_matrix = []
            '''


