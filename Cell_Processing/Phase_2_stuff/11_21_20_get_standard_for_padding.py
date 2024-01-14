'''
8/12/20
Program whose sole purpose is to determine what the standard of the padding should be for all the images.

At the end of the program, will print out the maximum x, maximum y, and maximum z to be used as a tuple
for the 3D matrices.

Will now also print out the average of the x, y and z values.
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

'''
def create_label_matrix(the_directory_name):
    for path in Path(starting_directory).rglob('*.tif'):
        if "accuracy" in path.name:
            print(path.name)
            x = str(path.name)
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
                            fibroblast_counter = fibroblast_counter + 1
                        else:
                            cancer_cell_counter = cancer_cell_counter + 1

    print("Total number of fibroblasts is " + str(fibroblast_counter))
    print("Total number of cancer cells is " + str(cancer_cell_counter))
'''

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)
def stack_and_crop(array_to_crop,array_to_stack_on):
    keeper_rows = []
    keeper_columns = []
    for y in range(len(array_to_crop)):
        for x in range(len(array_to_crop[0])):
            if array_to_crop[y][x] != 0:
                if not (y in keeper_rows):
                    keeper_rows.append(y)
                if not (x in keeper_columns):
                    keeper_columns.append(x)
    minimum_row = min(keeper_rows)
    minimum_column = min(keeper_columns)
    maximum_row = max(keeper_rows)
    maximum_column = max(keeper_columns)
    cropped_arr1 = np.zeros((maximum_row-minimum_row+1, maximum_column-minimum_column+1))
    counterY = 0
    counterX = 0
    for y in range(minimum_row, maximum_row + 1):
        for x in range(minimum_column, maximum_column + 1):
            cropped_arr1[counterY][counterX] = array_to_crop[y][x]
            counterX = counterX + 1
        counterX = 0
        counterY = counterY + 1
    counterY = 0
    counterX = 0
    #print(cropped_arr1)
    array_to_stack_on.append(cropped_arr1)
    return array_to_stack_on


def get_the_standard(the_list, dir):
    global true_maximum_z, true_maximum_x, true_maximum_y, directory_true_maximum_z, directory_true_maximum_y, \
        directory_true_maximum_x, sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values
    max_z = len(the_list)
    max_y = 0
    max_x = 0
    for z in range(len(the_list)):
        if len(the_list[z]) > max_y:
            max_y = len(the_list[z])
        for y in range(len(the_list[z])):
            if len(the_list[z][y]) > max_x:
                max_x = len(the_list[z][y])
    sum_of_x_values = sum_of_x_values + max_x
    sum_of_y_values = sum_of_y_values + max_y
    sum_of_z_values = sum_of_z_values + max_z
    num_values = num_values + 1
    if max_z > true_maximum_z:
        true_maximum_z = max_z
        directory_true_maximum_z = dir
    if max_y > true_maximum_y:
        true_maximum_y = max_y
        directory_true_maximum_y = dir
    if max_x > true_maximum_x:
        true_maximum_x = max_x
        directory_true_maximum_x = dir


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
def pad_matrices_in_list_to_standard(the_list, tuple_size):
    padding_matrix = np.zeros((tuple_size))
    for z in range(len(the_list)):
        for y in range(len(the_list[z])):
            for x in range(len(the_list[z][y])):
                padding_matrix[y][x] = the_list[z][y][x]
    return padding_matrix


starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X (no standard) Images Macro V5\step2 - 2D matrices"
#new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 4"
checkDirectory(starting_directory)
#checkDirectory(new_directory)

# The 5 3D arrays off this data
DAPI_3D_array = []
#Fibroblast_3D_array = []
#Cancer_3D_array = []
Reflection_3D_array = []
Transmission_brightfield_3D_array = []

true_maximum_z = 0
directory_true_maximum_z = ""
true_maximum_y = 0
directory_true_maximum_y = ""
true_maximum_x = 0
directory_true_maximum_x = ""

sum_of_x_values = 0
sum_of_y_values = 0
sum_of_z_values = 0
num_values = 0

for root, dirs, files in os.walk(starting_directory):
    for dir in dirs:
        # Begin searching each directory for the 2D PNGs
        print("Now searching directory " + str(dir))
        for path in Path(os.path.join(starting_directory,dir)).rglob("*.NPY"):
            x = str(path.name)
            if x[0] == "C":
                if x[1] == "1":
                    print("This path has a C1 image: " + path.name)
                    arr1 = np.load(str(path))
                    #print(arr1)
                    stack_and_crop(arr1,DAPI_3D_array)
                if x[1] == "2":
                    print("This path has a C2 image: " + path.name)
                    #arr2 = np.load(str(path))
                    #print(arr2)
                    #stack_and_crop(arr2, Fibroblast_3D_array)
                if x[1] == "3":
                    print("This path has a C3 image: " + path.name)
                    #arr3 = np.load(str(path))
                    #print(arr3)
                    #stack_and_crop(arr3, Cancer_3D_array)
                if x[1] == "4":
                    print("This path has a C4 image: " + path.name)
                    arr4 = np.load(str(path))
                    #print(arr4)
                    stack_and_crop(arr4, Reflection_3D_array)
                if x[1] == "5":
                    print("This path has a C5 image: " + path.name)
                    arr5 = np.load(str(path))
                    #print(arr5)
                    stack_and_crop(arr5, Transmission_brightfield_3D_array)

        # Padding the matrices
        get_the_standard(DAPI_3D_array, dir)
        #Fibroblast_3D_array = pad_matrices_in_list(Fibroblast_3D_array)
        #Cancer_3D_array = pad_matrices_in_list(Cancer_3D_array)
        get_the_standard(Reflection_3D_array, dir)
        get_the_standard(Transmission_brightfield_3D_array, dir)



        # reset the 3D matrices after saving them.
        DAPI_3D_array = []
        # Fibroblast_3D_array = []
        # Cancer_3D_array = []
        Reflection_3D_array = []
        Transmission_brightfield_3D_array = []

print("The true maximum z is " + str(true_maximum_z))
print("The directory where true maximum z is located is " + str(directory_true_maximum_z))
print("The true maximum y is " + str(true_maximum_y))
print("The directory where true maximum y is located is " + str(directory_true_maximum_y))
print("The true maximum x is " + str(true_maximum_x))
print("The directory where true maximum x is located is " + str(directory_true_maximum_x))

print("The average for the x values is " + str(sum_of_x_values / num_values))
print("The average for the y values is " + str(sum_of_y_values / num_values))
print("The average for the z values is " + str(sum_of_z_values / num_values))

'''
DAPI_3D_array = np.array(DAPI_3D_array)
Fibroblast_3D_array = np.array(Fibroblast_3D_array)
Cancer_3D_array = np.array(Cancer_3D_array)
Reflection_3D_array = np.array(Reflection_3D_array)
Transmission_brightfield_3D_array = np.array(Transmission_brightfield_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\DAPI_3D_array",DAPI_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Fibroblast_3D_array",Fibroblast_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Cancer_3D_array",Cancer_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Reflection_3D_array",Reflection_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Transmission_brightfield_3D_array",Transmission_brightfield_3D_array )
'''



'''keeper_rows = []
                    keeper_columns = []
                    for y in range(len(arr1)):
                        for x in range(len(arr1[0])):
                            if arr1[y][x] != 0:
                                if not (y in keeper_rows):
                                    keeper_rows.append(y)
                                if not (x in keeper_columns):
                                    keeper_columns.append(x)
                    cropped_arr1 = np.zeros(len(keeper_rows),len(keeper_columns))
                    minimum_row = min(keeper_rows)
                    minimum_column = min(keeper_columns)
                    maximum_row = max(keeper_rows)
                    maximum_column = max(keeper_columns)
                    for y in range(minimum_row,maximum_row+1):
                        for x in range(minimum_column,maximum_column+1):
                            cropped_arr1 = arr1[y][x]
                    print(cropped_arr1)
                    DAPI_3D_array.append(cropped_arr1)'''