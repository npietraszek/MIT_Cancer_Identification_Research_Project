import os
import numpy as np
from pathlib import Path
from stack_and_crop import stack_and_crop

'''
Support function for finding the 2D cell images of each image type inside a directory and stacking them into their corresponding 3D arrays.
Meant to be used inside an os.walk loop to go through all the images in a directory.
Does not stack the cancer cell and fibroblast cell matrices as those images are not used by the machine learning.
Parameters
----------
starting_directory : string
    The parent directory to find the cell directories in.
dir : string
    The directory currently being searched for cell images to stack.
DAPI_3D_array : numpy array
    The 3D array to stack the DAPI images into.
Reflection_3D_array : numpy array
    The 3D array to stack the reflection images into.
Transmission_brightfield_3D_array : numpy array
    The 3D array to stack the transmission brightfield images into.
test : boolean
    Whether or not to print out extra information for testing purposes.
Returns
-------
DAPI_3D_array : numpy array
    The 3D array with the DAPI images stacked into it.
Reflection_3D_array : numpy array
    The 3D array with the reflection images stacked into it.
Transmission_brightfield_3D_array : numpy array
    The 3D array with the transmission brightfield images stacked into it.
    \
'''
def image_os_walker(starting_directory, dir, DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array, test=False):
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
    return (DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array)