'''
8/11/20
Code that both crops out empty slices from the images
and saves the usable slices as 2D matrices inside seperate folders, with a folder for each 3D image.

This code is useful (and intended to be used alongside matrix_stacker_V3), but it might be better
to combine the functions of both in a bigger program.

'''




import os
import numpy as np
import PIL
import matplotlib.pyplot as plt
from pathlib import Path
from Common_Utils.checkDirectory import checkDirectory

'''
Function to remove empty slices from the 3D cell images, then save the remaining usable slices as 2D matrices inside seperate folders, 
with a folder for each 3D image.

Parameters
----------
starting_directory : string
    The directory to find the cell images in.
target_directory : string
    The directory to save the 2D matrices in.
test : boolean
    Whether or not to print out extra information for testing purposes.

Returns
-------
None.
'''
def cell_cropper(starting_directory, target_directory, test=False):
    plt.gray()
    # The directory to start searching in recursively.
    checkDirectory(starting_directory)
    checkDirectory(target_directory)

    for path in Path(starting_directory).rglob('*.PNG'):

        directory_structure = os.path.dirname(str(path))

        cell_directory = os.path.basename(directory_structure)

        if test == True:
            print("The full path is " + str(path))
            print("Directory structure is " + str(directory_structure))
            print("The folder we want is " + str(cell_directory))
        
        cloned_cell_folder = target_directory + "/" + cell_directory
        checkDirectory(cloned_cell_folder)
        img = PIL.Image.open(path)
        arr = np.array(img)
        isempty = True
        for y in range(len(arr)):
            for x in range(len(arr[0])):
                if arr[y][x] != 0:
                    isempty = False
                    if test == True:
                        print("(" + str(x) + "," + str(y) + ") has value " + str(arr[y][x]))
        if isempty == False:
            if test == True:
                print("Saving matrix " + str(path.name) + " at " + str(cloned_cell_folder + "/" + path.name))
            np.save(file = cloned_cell_folder + "/" + path.name,arr = arr)
            if test == True:
                # Save a 2D slice of the 3D matrix as a PNG to check that the cell has been cropped properly.
                plt.imsave(arr = arr, fname= cloned_cell_folder + "/" + path.name + "TEST_MATRIX.png")

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step1 TIFFs and PNGs"
    target_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step2 - 2D matrices"
    cell_cropper(starting_directory, target_directory)
