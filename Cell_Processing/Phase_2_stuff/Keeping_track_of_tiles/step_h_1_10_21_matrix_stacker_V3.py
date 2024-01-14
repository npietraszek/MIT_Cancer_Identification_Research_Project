'''
8/28/20
This program works alongside version 2 of the cell_cropper after an image size standard has been enforced.

Differences from version 2:

Now uses the image standard generated from get_standard_for_padding.py.
Now has proper directory variables

THINGS TO DO:
Make sure the array pads properly. May want to begin the process of finding a standard size.
Make sure the array crops properly.
Turn everything into one program? Skip saving into 2D matrices, and start with just the matrices and work up?

NEED TO DO SYSTEMIC TESTING WHEN ALL PROGRAMMING IS FINISHED:
1) Check to make sure the PNG to matrix transformation is credible (cell_cropper_V2) (DONE)
2) Make sure the empty slice cropping works (cell_cropper_V2) (DONE)
3) Make sure label matrices are created correctly (DONE)
4) Make sure cropping works (DONE?)
5) Make sure padding works properly
6) Make sure the stacking works properly (DONE)
7) Make sure matrix is saved in the correct places and the correct folders (DONE)

STEPS COMPLETED:
0) Use glob (rglob?) to find every cell image. (Need to find a way to work off path names, perhaps?) (DONE)
1) Turn PNGs (or other images) into matrices (DONE)
2) Crop any completely empty slices (DONE)
3) Create a label matrix for the proper cell type each 3D matrix has.
4) Look through and remember which rows and columns have information in them (are not null) across all 2D slices
in 3D matrix, and crop all other rows and columns out of the 2D slices of the 3D matrix.
5) Pad the matrices afterward so that they have a standard size. Might need to come back to this later when we have
a standard size for all the images.
6) Take the 2D matrices and turn them into a 3D matrix.
7) Save the cropped 3D matrix with the same name as the cell image name
8) Repeat this for every cell in the folder

Program runs:

This program was run on 6/28/20 on
starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\attempt 1 without any of the strange cells"
new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\3D_matrices_(no_strange_cells)"

This was the first attempt to stack and pad the matrices. Will probably come back later once we have properly curated images
to try this again.


'''





import numpy as np
from pathlib import Path
import os
from utils.checkDirectory import checkDirectory
from utils.image_os_walker import image_os_walker
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

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


# Modified to account for chip name at beginning of each folder
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
# Need some kind of padding function??? Need to figure out how to get these matrices into a standard shape...
def stack_matrices_in_list(the_list):
    stacked_matrix = np.zeros((len(the_list),len(the_list[0]),len(the_list[0][0])))
    for z in range(len(the_list)):
        for y in range(len(the_list[z])):
            for x in range(len(the_list[z][y])):
                stacked_matrix = the_list[z][y][x]
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
                        print("Fibroblast label matrix: " + str(label_matrix))
                        path_to_matrix = os.path.join(new_directory, dir, "label_matrix")
                        checkDirectory(path_to_matrix)
                        np.save(os.path.join(path_to_matrix, "Label_matrix"), label_matrix)
                    else:
                        # create cancer cell label matrix
                        # cancer_cell_counter = cancer_cell_counter + 1
                        label_matrix = np.asarray([0, 1])
                        print("Cancer label matrix: " + str(label_matrix))
                        path_to_matrix = os.path.join(new_directory, dir, "Label_matrix")
                        checkDirectory(path_to_matrix)
                        np.save(os.path.join(path_to_matrix, "Label_matrix"), label_matrix)

            '''
            for ii in range(len(y)):
                if y[ii] == "F":
                    if y[ii + 1] == "b":
                        letters_before_fb = y[:ii]
                        letters_after_fb = y[ii:]
                        print("letters before fb = " + str(letters_before_fb))
                        print("letters after fb = " + str(letters_after_fb))
                        list_of_numbers_before_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_before_fb)]
                        list_of_numbers_after_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_fb)]
                        # print("numbers before fb = " + str(list_of_numbers_before_fb))
                        # print("numbers after fb = " + str(list_of_numbers_after_fb))
                        fibroblast_number = list_of_numbers_before_fb[-1]
                        cancer_cell_number = list_of_numbers_after_fb[0]
                        print("fibroblast number = " + str(fibroblast_number))
                        print("cancer cell number = " + str(list_of_numbers_after_fb[0]))
                        if fibroblast_number > cancer_cell_number:
                            # create fibroblast label matrix
                            # fibroblast_counter = fibroblast_counter + 1
                            label_matrix = np.asarray([1,0])
                            print("Fibroblast label matrix: " + str(label_matrix))
                            path_to_matrix = os.path.join(new_directory, dir, "label_matrix")
                            checkDirectory(path_to_matrix)
                            np.save(os.path.join(path_to_matrix,"Label_matrix"),label_matrix)
                        else:
                            # create cancer cell label matrix
                            # cancer_cell_counter = cancer_cell_counter + 1
                            label_matrix = np.asarray([0, 1])
                            print("Cancer label matrix: " + str(label_matrix))
                            path_to_matrix = os.path.join(new_directory, dir,"Label_matrix")
                            checkDirectory(path_to_matrix)
                            np.save(os.path.join(path_to_matrix,"Label_matrix"),label_matrix)
            '''
            # Begin searching each directory for the 2D PNGs
            print("Now searching directory " + str(dir))
            DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array = image_os_walker(starting_directory, dir, DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array, test=test)
            '''
            for path in Path(os.path.join(starting_directory,dir)).rglob("*.NPY"):
                x = str(path.name)
                if x[0] == "C":
                    if x[1] == "1":
                        print("This path has a C1 image: " + path.name)
                        arr1 = np.load(str(path))
                        # print(arr1)
                        stack_and_crop(arr1,DAPI_3D_array)
                    if x[1] == "2":
                        print("This path has a C2 image: " + path.name)
                        # arr2 = np.load(str(path))
                        # print(arr2)
                        # stack_and_crop(arr2, Fibroblast_3D_array)
                    if x[1] == "3":
                        print("This path has a C3 image: " + path.name)
                        # arr3 = np.load(str(path))
                        # print(arr3)
                        # stack_and_crop(arr3, Cancer_3D_array)
                    if x[1] == "4":
                        print("This path has a C4 image: " + path.name)
                        arr4 = np.load(str(path))
                        # print(arr4)
                        stack_and_crop(arr4, Reflection_3D_array)
                    if x[1] == "5":
                        print("This path has a C5 image: " + path.name)
                        arr5 = np.load(str(path))
                        # print(arr5)
                        stack_and_crop(arr5, Transmission_brightfield_3D_array)
            '''

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