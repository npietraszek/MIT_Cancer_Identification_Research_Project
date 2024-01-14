import os
import numpy as np
from pathlib import Path
from stack_and_crop import stack_and_crop


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