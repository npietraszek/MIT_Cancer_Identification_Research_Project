'''
8/28/20

NOTE: Has been improved by

This program uses the matrices generated from matrix_stacker_V3.py and stacks the 3D matrices into 5D ones.
It works by stacking the corresponding DAPI, brightfield and reflective matrices into a 5D matrix in a
1 by 3 by Z by Y by X.


Steps:

1) Load the 3D matrices with the correct names for each image type.
1a) Load the label matrix and store it separately
2) Stack the 3D matrices onto the 5D matrix in the exactly correct dimensions.
3) Save the 5D matrix and the label matrix together in the same directory.

Debugging:
Correctly identifies the matrices in directories.
Seems to work properly in creating and saving the 5D matrices and the label matrices.

NOTE: Might need to come back to this to change the shape of the 5D matrices as they are generated.
I don't trust np.reshape at all...

EDIT: Use tf.permute...

In the 6/28/20 edition, Z = 20, Y = 30, and X = 30.

Now uses the image standard generated from get_standard_for_padding.py

Saves DAPI to [0], Reflective to [1], and Brightfield to [2].
'''




import os
import numpy as np
from pathlib import Path
from Common_Utils.checkDirectory import checkDirectory
from utils.image_os_walker import image_os_walker

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


def stack_3D_into_5D(starting_directory, new_directory, shape_of_3D_matrices):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)
    # The 3 3D arrays off this data
    DAPI_3D_array = []
    Reflection_3D_array = []
    Transmission_brightfield_3D_array = []

    # The arrays to be finally saved
    The_final_5D_array = np.zeros((3,shape_of_3D_matrices[0],shape_of_3D_matrices[1],shape_of_3D_matrices[2]))
    The_label_matrix = []


    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            # Begin searching each directory for the 2D PNGs
            if dir != "DAPI_3D_array" and dir != "Label_matrix" and dir != "Reflection_3D_array" and dir != "Transmission_brightfield_3D_array" and dir != "label_matrix":
                print("Now searching directory " + str(dir))
                for path in Path(os.path.join(starting_directory,dir)).rglob("*.NPY"):
                    x = str(path.name)
                    if x[0] == "D":
                        if x[1] == "A":
                            if x[2] == "P":
                                if x[3] == "I":
                                    print("This path has a DAPI 3D matrix: " + path.name)
                                    arr1 = np.load(str(path))
                                    DAPI_3D_array = arr1[:]
                                    print("Finished")
                    if x[0] == "T":
                        if x[1] == "r":
                            if x[2] == "a":
                                if x[3] == "n":
                                    if x[4] == "s":
                                        print("This path has a Transmission_brightfield_3D_array: " + path.name)
                                        arr2 = np.load(str(path))
                                        Transmission_brightfield_3D_array = arr2[:]
                                        print("Finished")
                    if x[0] == "R":
                        if x[1] == "e":
                            if x[2] == "f":
                                if x[3] == "l":
                                    print("This path has a Reflection_3D_array: " + path.name)
                                    arr3 = np.load(str(path))
                                    Reflection_3D_array = arr3[:]
                                    print("Finished")
                    if x[0] == "L":
                        if x[1] == "a":
                            if x[2] == "b":
                                if x[3] == "e":
                                    print("This path has a label matrix: " + path.name)
                                    arr4 = np.load(str(path))
                                    The_label_matrix = arr4[:]
                                    print("Finished")


                            '''
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
                        '''

                # After padding, the lists can properly be turned into matrices.
                DAPI_3D_array = np.array(DAPI_3D_array)
                Reflection_3D_array = np.array(Reflection_3D_array)
                Transmission_brightfield_3D_array = np.array(Transmission_brightfield_3D_array)
                The_label_matrix = np.array(The_label_matrix)

                The_final_5D_array = stack_DAPI_into_5D(DAPI_3D_array,The_final_5D_array)
                The_final_5D_array = stack_Reflective_into_5D(Reflection_3D_array,The_final_5D_array)
                The_final_5D_array = stack_Transmission_brightfield_into_5D(Transmission_brightfield_3D_array,The_final_5D_array)


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


if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step4 3D matrices"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step5 4D matrices"
    shape_of_3D_matrices = [20,50,50]
    stack_3D_into_5D(starting_directory, new_directory, shape_of_3D_matrices)