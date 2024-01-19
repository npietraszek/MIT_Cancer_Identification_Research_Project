'''
8/28/20
The most complex part of the preprocessing pipeline. This program takes the non-strange 2D images from the cell_cropper, pads them according'
to the image standard (20-50-50), creates label matrices associated with them, and saves them as 3D matrices.


'''





import numpy as np
from pathlib import Path
import os
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.image_os_walker import image_os_walker
from Common_Utils.splitall import splitall

'''
Pads the matrices in the list based on the maximum length and width of the matrices in the list.
'''
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


'''
Finds the original cell ROI folder for a given cell folder in order to find the ground truth label matrix associated with it.
Modified to account for chip name at beginning of each folder.
Parameters
----------
cell_folder_name : string
    The name of the cell folder to find the original cell ROI folder for.
ground_truths_directory : string
    The directory to search for the original cell ROI folder in.
Returns
-------
final_folder : string
    The directory of the original cell ROI folder.
'''
def find_original_cell_ROI(cell_folder_name, ground_truths_directory):
    final_folder = r""
    if "device 1 chip 1 and 2" in cell_folder_name:
        cut_folder_name = cell_folder_name[22:]
    elif "device 1 chip 3" in cell_folder_name:
        cut_folder_name = cell_folder_name[16:]
    elif "device 2" in cell_folder_name:
        cut_folder_name = cell_folder_name[9:]
    elif "device 3 chip 1 2 3" in cell_folder_name:
        cut_folder_name = cell_folder_name[20:]
    elif "device 3 chip 3" in cell_folder_name:
        cut_folder_name = cell_folder_name[16:]
    else:
        raise ValueError("device identification not found")
    for path in Path(ground_truths_directory).rglob("*.tif"):
        #print(path)
        path_parts = splitall(path)
        the_name = os.path.splitext(path_parts[-1])[0]
        if the_name == cut_folder_name:
            final_folder = os.path.split(path)[0]
            return final_folder

    print("Not found")
    raise ValueError("Original cell ROI not found")




'''
Main function of the program. 
Takes the 2D images from the cell_cropper, pads them according to the image standard (20-50-50), 
creates label matrices associated with them, and saves them as 3D matrices.
Parameters
----------
ground_truths_directory : string
    The directory to find the ground truth label matrices in.
starting_directory : string
    The directory to find the 2D images in.
new_directory : string  
    The directory to save the 3D matrices in.
test : boolean
    Whether or not to print out extra information for testing purposes.
Returns
-------
None.
'''
def matrix_stacker_V3(ground_truths_directory, starting_directory, new_directory, test=False):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)

    # The 5 3D arrays off this data
    DAPI_3D_array = []
    #Fibroblast_3D_array = []
    #Cancer_3D_array = []
    Reflection_3D_array = []
    Transmission_brightfield_3D_array = []


    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            # Make a label matrix for the designated folder
            y = str(dir)
            ground_truth_folder = find_original_cell_ROI(dir, ground_truths_directory)
            for path in Path(ground_truth_folder).rglob('*.tif'):
                if "groundTruth" in path.name:
                    if "fibroblast" in path.name:
                        # create fibroblast label matrix
                        # fibroblast_counter = fibroblast_counter + 1
                        label_matrix = np.asarray([1, 0])
                        if test == True:
                            print("Fibroblast label matrix: " + str(label_matrix))
                        path_to_matrix = os.path.join(new_directory, dir, "label_matrix")
                        checkDirectory(path_to_matrix)
                        np.save(os.path.join(path_to_matrix, "Label_matrix"), label_matrix)
                    else:
                        # create cancer cell label matrix
                        # cancer_cell_counter = cancer_cell_counter + 1
                        label_matrix = np.asarray([0, 1])
                        if test == True:
                            print("Cancer label matrix: " + str(label_matrix))
                        path_to_matrix = os.path.join(new_directory, dir, "Label_matrix")
                        checkDirectory(path_to_matrix)
                        np.save(os.path.join(path_to_matrix, "Label_matrix"), label_matrix)

            # Begin searching each directory for the 2D PNGs
            if test == True:
                print("Now searching directory " + str(dir))
            DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array = image_os_walker(starting_directory, dir, DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array, test=test)

            # Padding the matrices
            DAPI_3D_array = pad_matrices_in_list_to_standard(DAPI_3D_array,20,50,50)
            # Fibroblast_3D_array = pad_matrices_in_list(Fibroblast_3D_array)
            # Cancer_3D_array = pad_matrices_in_list(Cancer_3D_array)
            Reflection_3D_array = pad_matrices_in_list_to_standard(Reflection_3D_array,20,50,50)
            Transmission_brightfield_3D_array = pad_matrices_in_list_to_standard(Transmission_brightfield_3D_array,20,50,50)

            # After padding, the lists can properly be turned into matrices.
            DAPI_3D_array = np.array(DAPI_3D_array)
            Reflection_3D_array = np.array(Reflection_3D_array)
            Transmission_brightfield_3D_array = np.array(Transmission_brightfield_3D_array)


            DAPI_path = os.path.join(new_directory, dir, r"DAPI_3D_array")
            reflection_path = os.path.join(new_directory, dir,r"Reflection_3D_array")
            transmission_path = os.path.join(new_directory, dir, r"Transmission_brightfield_3D_array")

            checkDirectory(DAPI_path)
            checkDirectory(reflection_path)
            checkDirectory(transmission_path)

            np.save(os.path.join(DAPI_path ,r"DAPI_3D_array"), DAPI_3D_array)
            np.save(os.path.join(reflection_path,r"Reflection_3D_array"),Reflection_3D_array)
            np.save(os.path.join(transmission_path,r"Transmission_brightfield_3D_array"), Transmission_brightfield_3D_array)

            # reset the 3D matrices after saving them.
            DAPI_3D_array = []
            Reflection_3D_array = []
            Transmission_brightfield_3D_array = []


if __name__ == "__main__":
    ground_truths_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V5 Macro"
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step3 2D image standard"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step4 3D matrices"
    matrix_stacker_V3(ground_truths_directory, starting_directory, new_directory, test=False)