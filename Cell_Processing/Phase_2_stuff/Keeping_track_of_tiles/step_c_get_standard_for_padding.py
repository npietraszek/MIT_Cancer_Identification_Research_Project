'''
8/12/20
Program whose sole purpose is to determine what the standard of the padding should be for all the images.

Since the machine learning can only work with a single matrix size, all of the cell images need to be standardized to a single size.
In order to make sure no information is lost, we need to find the maximum x, y and z values of all the 3D matrices after the cell_cropper finishes.

Hence, this program is used to find the maximum x, y and z values of all the 3D matrices after the cell_cropper finishes.
It also gathers the directory where the maximum dimensions are found, as well as the average cell size for diagnostic purposes.
'''




import os
import numpy as np
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.image_os_walker import image_os_walker

'''
Collects information about the total dimensions of the 3D cell images for a particular directory.
Parameters
----------
the_matrix: numpy matrix
    The 3D matrix to get the dimensions of.
dir: string
    The directory the 3D matrix is located in.
sum_of_x_values: int
    The sum of all the x values of the 3D matrices. Diagnostic value.
sum_of_y_values: int
    The sum of all the y values of the 3D matrices. Diagnostic value.
sum_of_z_values: int
    The sum of all the z values of the 3D matrices. Diagnostic value.
num_values: int
    The number of 3D matrices that have been processed. Diagnostic value.
true_maximum_x: int
    The current maximum x value of all the 3D matrices. Needed for the padding standard.
true_maximum_y: int
    The current maximum y value of all the 3D matrices. Needed for the padding standard.
true_maximum_z: int
    The current maximum z value of all the 3D matrices. Needed for the padding standard.
directory_true_maximum_x: string
    The directory where the current maximum x value of all the 3D matrices is located. Diagnostic value.
directory_true_maximum_y: string
    The directory where the current maximum y value of all the 3D matrices is located. Diagnostic value.
directory_true_maximum_z: string
    The directory where the current maximum z value of all the 3D matrices is located. Diagnostic value.
Returns
-------
sum_of_x_values: int
    The sum of all the x values of the 3D matrices. Diagnostic value.
sum_of_y_values: int
    The sum of all the y values of the 3D matrices. Diagnostic value.
sum_of_z_values: int
    The sum of all the z values of the 3D matrices. Diagnostic value.
num_values: int
    The number of 3D matrices that have been processed. Diagnostic value.
true_maximum_x: int
    The current maximum x value of all the 3D matrices. Needed for the padding standard.
true_maximum_y: int
    The current maximum y value of all the 3D matrices. Needed for the padding standard.
true_maximum_z: int
    The current maximum z value of all the 3D matrices. Needed for the padding standard.
directory_true_maximum_x: string
    The directory where the current maximum x value of all the 3D matrices is located. Diagnostic value.
directory_true_maximum_y: string
    The directory where the current maximum y value of all the 3D matrices is located. Diagnostic value.
directory_true_maximum_z: string
    The directory where the current maximum z value of all the 3D matrices is located. Diagnostic value.

'''
def get_the_standard_for_directory(the_matrix, dir, sum_of_x_values, sum_of_y_values, sum_of_z_values, num_values, true_maximum_x, true_maximum_y, true_maximum_z, directory_true_maximum_x, directory_true_maximum_y, directory_true_maximum_z):
    max_z = len(the_matrix)
    max_y = 0
    max_x = 0
    for z in range(len(the_matrix)):
        if len(the_matrix[z]) > max_y:
            max_y = len(the_matrix[z])
        for y in range(len(the_matrix[z])):
            if len(the_matrix[z][y]) > max_x:
                max_x = len(the_matrix[z][y])
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

'''
Function to find the maximum x, y and z values of all the 3D matrices after the cell_cropper finishes.
Parameters
----------
starting_directory : string
    The directory to start searching in recursively.
test : boolean
    Whether or not to print out extra information for testing purposes.
Returns 
-------
true_maximum_x : int
    The maximum x value of all the 3D matrices.
directory_true_maximum_x : string
    The directory where the maximum x value of all the 3D matrices is located. Diagnostic value.
true_maximum_y : int
    The maximum y value of all the 3D matrices.
directory_true_maximum_y : string   
    The directory where the maximum y value of all the 3D matrices is located. Diagnostic value.
true_maximum_z : int
    The maximum z value of all the 3D matrices.
directory_true_maximum_z : string
    The directory where the maximum z value of all the 3D matrices is located. Diagnostic value.
sum_of_x_values : int
    The sum of all the x values of the 3D matrices. Diagnostic value.
sum_of_y_values : int
    The sum of all the y values of the 3D matrices. Diagnostic value.
sum_of_z_values : int
    The sum of all the z values of the 3D matrices. Diagnostic value.
'''
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
