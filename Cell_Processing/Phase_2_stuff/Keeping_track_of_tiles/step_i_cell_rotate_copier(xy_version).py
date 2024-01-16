'''
7/2/20

VERY IMPORTANT TOOL. TESTED EXTENSIVELY
Multiplies the number of matrices in a folder by 4 by rotating all the images on the xy axis
in the folder by 90,180, and then 270 degrees, then saves them as separate images in a separate folder.

This should increase the number of images and the accuracy we have to work with.

The xy axis rotation has been tested extensively and has no problem.

'''




import os
import numpy as np
from pathlib import Path
from Common_Utils.checkDirectory import checkDirectory

def cell_rotate_copier(starting_directory, target_directory):
    checkDirectory(starting_directory)
    checkDirectory(target_directory)

    fibroblast_counter = 0
    cancer_cell_counter = 0
    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            x = str(dir)
            for path in Path(os.path.join(starting_directory,dir)).rglob("*.NPY"):
                y = str(path.name)
                if y[0] == "F":
                    if y[1] == "i":
                        if y[2] == "n":
                            #It's a final 5D array!
                            arr1 = np.load(str(path))

                            # Rotation methods. Tried and true for xy.
                            final_array_1 = arr1[:]
                            final_array_2 = np.rot90(final_array_1,1, (2,3))
                            final_array_3 = np.rot90(final_array_1,2, (2,3))
                            final_array_4 = np.rot90(final_array_1,3, (2,3))
                            destination_1 = os.path.join(target_directory, dir, "Final_5D_array_1.npy")
                            destination_2 = os.path.join(target_directory, dir, "Final_5D_array_2.npy")
                            destination_3 = os.path.join(target_directory, dir, "Final_5D_array_3.npy")
                            destination_4 = os.path.join(target_directory, dir, "Final_5D_array_4.npy")
                            checkDirectory(os.path.join(target_directory,dir))
                            np.save(str(destination_1),final_array_1)
                            np.save(str(destination_2), final_array_2)
                            np.save(str(destination_3), final_array_3)
                            np.save(str(destination_4), final_array_4)

                            # Try loading the file you just made
                            test_matrix_1 = np.load(destination_1)
                            test_matrix_2 = np.load(destination_2)
                            test_matrix_3 = np.load(destination_3)
                            test_matrix_4 = np.load(destination_4)
                if y[0] == "L":
                    if y[1] == "a":
                        if y[2] == "b":
                            # It's a label matrix!
                            arr1 = np.load(str(path))
                            label_array = arr1[:]
                            destination = os.path.join(target_directory,dir,"Label_matrix.npy")
                            np.save(str(destination),label_array)

                            # Try loading the file you just made
                            test_matrix = np.load(destination)


if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step6 Balanced 4D matrices"
    target_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step7 Rotated 4D matrices"
    cell_rotate_copier(starting_directory, target_directory)