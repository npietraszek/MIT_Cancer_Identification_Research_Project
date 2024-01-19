# This file is used to sort the matrices into their tiles based on the information in their folder names.



import os
import shutil
from Common_Utils.checkDirectory import checkDirectory


'''
Sorts the matrices into their tiles based on the information in their folder names.
Afterward, writes a file with the number of cells in each ROI tile.
Is needed in order to do the 3D reconstruction to demonstrate the machine learning's accuracy for the 1600 cells.
The 1600 cells all need to come from the same sets of tiles. Otherwise, when we reconstruct the 3D image, there will be a mix of 
training, validation, and testing cells in the same image, which would make the 3D reconstruction 
not a proper representation of the machine learning's accuracy.

Parameters
----------
starting_directory : str
    The directory to start searching in recursively.
new_directory : str
    The directory to copy the sorted cells into their respective tile folders.
Returns
-------
None
'''
def tile_seperator_and_counter(starting_directory, new_directory):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)

    dictionary_of_file_counters = {}
    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            x = str(dir)
            if "device 1 chip 1 and 2" in x:
                # 12 characters in ROI signification
                # [21:33]
                ROI_designation = x[21:33]
                if x[0:33] in dictionary_of_file_counters.keys():
                    dictionary_of_file_counters[x[0:33]] = dictionary_of_file_counters[x[0:33]] + 1
                else:
                    dictionary_of_file_counters[x[0:33]] = 1
                checkDirectory(os.path.join(new_directory,x[0:33]))

                path_to_original_matrix = os.path.join(starting_directory,x)
                path_to_new_matrix = os.path.join(new_directory,x[0:33],x)
                shutil.copytree(path_to_original_matrix,path_to_new_matrix)
            elif "device 1 chip 3" in x:
                # ROI3_01.oib
                # ROI5_01-1
                # doesn't matter if it's 9 characters, still unique
                ROI_designation = x[15:27]
                if x[0:27] in dictionary_of_file_counters.keys():
                    dictionary_of_file_counters[x[0:27]] = dictionary_of_file_counters[x[0:27]] + 1
                else:
                    dictionary_of_file_counters[x[0:27]] = 1
                checkDirectory(os.path.join(new_directory,x[0:27]))

                path_to_original_matrix = os.path.join(starting_directory,x)
                path_to_new_matrix = os.path.join(new_directory,x[0:27],x)
                shutil.copytree(path_to_original_matrix,path_to_new_matrix)
            elif "device 2" in x:
                # ROI3_01.oib
                # ROI5_01-1
                # doesn't matter if it's 9 characters, still unique
                ROI_designation = x[8:20]
                if x[0:20] in dictionary_of_file_counters.keys():
                    dictionary_of_file_counters[x[0:20]] = dictionary_of_file_counters[x[0:20]] + 1
                else:
                    dictionary_of_file_counters[x[0:20]] = 1
                checkDirectory(os.path.join(new_directory,x[0:20]))

                path_to_original_matrix = os.path.join(starting_directory,x)
                path_to_new_matrix = os.path.join(new_directory,x[0:20],x)
                shutil.copytree(path_to_original_matrix,path_to_new_matrix)
            elif "device 3 chip 1 2 3" in x:
                # ROI3_01.oib
                # ROI5_01-1
                # doesn't matter if it's 9 characters, still unique
                ROI_designation = x[19:31]
                if x[0:31] in dictionary_of_file_counters.keys():
                    dictionary_of_file_counters[x[0:31]] = dictionary_of_file_counters[x[0:31]] + 1
                else:
                    dictionary_of_file_counters[x[0:31]] = 1
                checkDirectory(os.path.join(new_directory, x[0:31]))

                path_to_original_matrix = os.path.join(starting_directory, x)
                path_to_new_matrix = os.path.join(new_directory, x[0:31], x)
                shutil.copytree(path_to_original_matrix, path_to_new_matrix)
            elif "device 3 chip 3" in x:
                # ROI3_01.oib
                # ROI5_01-1
                # doesn't matter if it's 9 characters, still unique
                ROI_designation = x[15:27]
                if x[0:27] in dictionary_of_file_counters.keys():
                    dictionary_of_file_counters[x[0:27]] = dictionary_of_file_counters[x[0:27]] + 1
                else:
                    dictionary_of_file_counters[x[0:27]] = 1
                checkDirectory(os.path.join(new_directory, x[0:27]))

                path_to_original_matrix = os.path.join(starting_directory, x)
                path_to_new_matrix = os.path.join(new_directory, x[0:27], x)
                shutil.copytree(path_to_original_matrix, path_to_new_matrix)
            else:
                raise ValueError("Device number not found")

    for key in dictionary_of_file_counters:
        print("ROI folder " + str(key) + " has " + str(dictionary_of_file_counters[key]) + " folders in it.")
        the_path_to_iterate = os.path.join(new_directory,key,"Number_of_cells_in_ROIOI.txt")
        print(str(the_path_to_iterate))
        file1 = open(the_path_to_iterate, "a")
        file1.write(str(dictionary_of_file_counters[key]))
        file1.close()
        #num_cells = len(next(os.walk(the_path_to_iterate))[1])

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step7 Rotated 4D matrices"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step8 Rotated ROI sorted 4D"
    tile_seperator_and_counter(starting_directory,new_directory)