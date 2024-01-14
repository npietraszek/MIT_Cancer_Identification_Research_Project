'''
8/12/20
Program whose sole purpose is to determine what the standard of the padding should be for all the images.

At the end of the program, will print out the maximum x, maximum y, and maximum z to be used as a tuple
for the 3D matrices.

Will now also print out the average of the x, y and z values.
'''




import os
import numpy as np
from pathlib import Path
from Common_Utils.checkDirectory import checkDirectory
from utils.image_os_walker import image_os_walker



'''
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
'''
def get_the_standard_for_directory(the_list, dir, sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values, true_maximum_x, true_maximum_y, true_maximum_z, directory_true_maximum_x, directory_true_maximum_y, directory_true_maximum_z):
    max_z = len(the_list)
    max_y = 0
    max_x = 0
    for z in range(len(the_list)):
        if len(the_list[z]) > max_y:
            max_y = len(the_list[z])
        for y in range(len(the_list[z])):
            if len(the_list[z][y]) > max_x:
                max_x = len(the_list[z][y])
    # Adjust values for the standard size.
    num_values = num_values + 1
    sum_of_x_values = sum_of_x_values + max_x
    sum_of_y_values = sum_of_y_values + max_y
    sum_of_z_values = sum_of_z_values + max_z
    if max_z > true_maximum_z:
        true_maximum_z = max_z
        directory_true_maximum_z = dir
    if max_y > true_maximum_y:
        true_maximum_y = max_y
        directory_true_maximum_y = dir
    if max_x > true_maximum_x:
        true_maximum_x = max_x
        directory_true_maximum_x = dir

    return (sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values, true_maximum_x, true_maximum_y, true_maximum_z, directory_true_maximum_x, directory_true_maximum_y, directory_true_maximum_z)


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

def get_standard_for_padding(starting_directory, test=False):
    checkDirectory(starting_directory)

    # The 3 3D arrays off this data
    # Fibroblast and cancer data are left off because the machine learning won't actually be using this data.
    DAPI_3D_array = []
    Reflection_3D_array = []
    Transmission_brightfield_3D_array = []

    # The true maximum values for the x, y and z dimensions, and the directories they are stored in
    # Used to determine what image standard we will apply.
    true_maximum_z = 0
    directory_true_maximum_z = ""
    true_maximum_y = 0
    directory_true_maximum_y = ""
    true_maximum_x = 0
    directory_true_maximum_x = ""

    # Some diagnostic values for tracking the average x, y and z values.
    sum_of_x_values = 0
    sum_of_y_values = 0
    sum_of_z_values = 0
    num_values = 0
    
    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            # Begin searching each directory for the 2D PNGs
            print("Now searching directory " + str(dir))
            DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array = image_os_walker(starting_directory, dir, DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array, test=test)
            '''
            for path in Path(os.path.join(starting_directory,dir)).rglob("*.NPY"):
                x = str(path.name)
                if x[0] == "C":
                    if x[1] == "1":
                        arr1 = np.load(str(path))
                        if test == True:
                            print("This path has a C1 (DAPI) image: " + path.name)
                            print(arr1)
                        stack_and_crop(arr1,DAPI_3D_array)
                    if x[1] == "2":
                        if test == True:
                            print("This path has a C2 (fibroblast) image: " + path.name)
                    if x[1] == "3":
                        if test == True:
                            print("This path has a C3 (cancer) image: " + path.name)
                    if x[1] == "4":
                        arr4 = np.load(str(path))
                        if test == True:
                            print("This path has a C4 (reflection) image: " + path.name)
                            print(arr4)
                        stack_and_crop(arr4, Reflection_3D_array)
                    if x[1] == "5":
                        arr5 = np.load(str(path))
                        if test == True:
                            print("This path has a C5 (Transmission_Brightfield) image: " + path.name)
                            print(arr5)
                        stack_and_crop(arr5, Transmission_brightfield_3D_array)
            '''
            # Padding the matrices
            get_the_standard_for_directory(DAPI_3D_array, dir, sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values, true_maximum_x, true_maximum_y, true_maximum_z, directory_true_maximum_x, directory_true_maximum_y, directory_true_maximum_z)
            get_the_standard_for_directory(Reflection_3D_array, dir, sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values, true_maximum_x, true_maximum_y, true_maximum_z, directory_true_maximum_x, directory_true_maximum_y, directory_true_maximum_z)
            get_the_standard_for_directory(Transmission_brightfield_3D_array, dir, sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values, true_maximum_x, true_maximum_y, true_maximum_z, directory_true_maximum_x, directory_true_maximum_y, directory_true_maximum_z)


            # reset the 3D matrices after saving them.
            DAPI_3D_array = []
            Reflection_3D_array = []
            Transmission_brightfield_3D_array = []
    if test == True:
        print("The true maximum x is " + str(true_maximum_x))
        print("The directory where true maximum x is located is " + str(directory_true_maximum_x))
        print("The true maximum y is " + str(true_maximum_y))
        print("The directory where true maximum y is located is " + str(directory_true_maximum_y))
        print("The true maximum z is " + str(true_maximum_z))
        print("The directory where true maximum z is located is " + str(directory_true_maximum_z))
        print("The average for the x values is " + str(sum_of_x_values / num_values))
        print("The average for the y values is " + str(sum_of_y_values / num_values))
        print("The average for the z values is " + str(sum_of_z_values / num_values))
    return (true_maximum_x, directory_true_maximum_x, true_maximum_y, directory_true_maximum_y, true_maximum_z, directory_true_maximum_z, sum_of_x_values / num_values, sum_of_y_values / num_values, sum_of_z_values / num_values)

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step3 2D image standard"
    get_standard_for_padding(starting_directory, test=False)
