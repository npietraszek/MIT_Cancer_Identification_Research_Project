'''
8/11/20
Code that both crops out empty slices from the images
and saves the usable slices as 2D matrices inside seperate folders, with a folder for each 3D image.

This code is useful (and intended to be used alongside matrix_stacker_V3), but it might be better
to combine the functions of both in a bigger program.

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

random.seed(version=2)
plt.gray()
def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)

# The directory to start searching in recursively.
starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\Macro V5 images\20X (no standard) Images Macro V5\step1 TIFFs and PNGs"
new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\Macro V5 images\20X (no standard) Images Macro V5\step2 - 2D matrices"
checkDirectory(starting_directory)
checkDirectory(new_directory)

for path in Path(starting_directory).rglob('*.PNG'):
    print("The full path is " + str(path))
    directory_structure = os.path.dirname(str(path))
    print("Directory structure is " + str(directory_structure))
    cell_directory = os.path.basename(directory_structure)
    print("The folder we want is " + str(cell_directory))
    cloned_cell_folder = new_directory + "/" + cell_directory
    checkDirectory(cloned_cell_folder)
    img = PIL.Image.open(path)
    arr = np.array(img)
    #print(arr)
    isempty = True
    for y in range(len(arr)):
        for x in range(len(arr[0])):
            if arr[y][x] != 0:
                isempty = False
                print("(" + str(x) + "," + str(y) + ") has value " + str(arr[y][x]))
    if isempty == False:
        print("Saving matrix " + str(path.name) + " at " + str(cloned_cell_folder + "/" + path.name))
        np.save(file = cloned_cell_folder + "/" + path.name,arr = arr)
        #plt.imsave(arr = arr, fname= cloned_cell_folder + "/" + path.name + "TEST_MATRIX.png")