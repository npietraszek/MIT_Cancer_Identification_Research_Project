'''
7/7/20

Program to take 4D matrices already generated and cut the brightfield image type out of them, then save them to
a seperate folder with the correct directory name...

Will specialize this for the xy rotated matrices only since those are the ones that count...

Debugging:
Correctly loads the correct number of directories from the right place: True
Correctly loads the matrices: True
Crops the correct data slice that we wanted to crop away (the brightfield): Seems to be True
Correctly saves the data matrices: True
Correctly saves the label matrices: True

Program works as intended!
'''




import os
import numpy as np
import shutil
from Common_Utils import checkDirectory

def remove_image_type_from_4D(array_to_remove_from,image_type):
    if image_type != "brightfield" and image_type != "reflective" and image_type != "DAPI": 
        raise ValueError("Invalid image type")
    new_array = np.zeros((2,20,50,50))
    for b in range(len(array_to_remove_from[0])):
        for c in range(len(array_to_remove_from[0][0])):
            for d in range(len(array_to_remove_from[0][0][0])):
                if image_type == "brightfield":
                    new_array[0][b][c][d] = array_to_remove_from[0][b][c][d]
                    new_array[1][b][c][d] = array_to_remove_from[1][b][c][d]
                elif image_type == "reflective":
                    new_array[0][b][c][d] = array_to_remove_from[0][b][c][d]
                    new_array[1][b][c][d] = array_to_remove_from[2][b][c][d]
                elif image_type == "DAPI":
                    new_array[0][b][c][d] = array_to_remove_from[1][b][c][d]
                    new_array[1][b][c][d] = array_to_remove_from[2][b][c][d]
    return new_array


def cut_single_image_type(starting_directory, target_directory, image_type, test = False):
    checkDirectory(starting_directory)
    checkDirectory(target_directory)

    list_of_directories_to_walk = next(os.walk(starting_directory))[1]
    for directory in list_of_directories_to_walk:
        num_cells_file = open(os.path.join(starting_directory, directory,  "Number_of_cells_in_ROIOI.txt"))
        num_cells = num_cells_file.read()
        num_cells_file.close()
        checkDirectory(os.path.join(target_directory, directory))
        file1 = open(os.path.join(target_directory, directory, r"Number_of_cells_in_ROIOI.txt"), "a")  # append mode
        file1.write(num_cells)
        file1.close()
        print("Number of cells file written")


        for root, dirs, files in os.walk(os.path.join(starting_directory,directory)):
            for dir in dirs:
                # Begin searching each directory for the 2D PNGs
                print("Now searching directory " + str(os.path.join(starting_directory,directory,dir)))
                final_array_1 = np.load(os.path.join(starting_directory, directory, dir, "Final_5D_array_1.npy"))
                final_array_2 = np.load(os.path.join(starting_directory, directory, dir, "Final_5D_array_2.npy"))
                final_array_3 = np.load(os.path.join(starting_directory, directory, dir, "Final_5D_array_3.npy"))
                final_array_4 = np.load(os.path.join(starting_directory, directory, dir,"Final_5D_array_4.npy"))
                label_array = np.load(os.path.join(starting_directory, directory, dir, "Label_matrix.npy"))


                new_array_1 = remove_image_type_from_4D(final_array_1, image_type)
                new_array_2 = remove_image_type_from_4D(final_array_2, image_type)
                new_array_3 = remove_image_type_from_4D(final_array_3, image_type)
                new_array_4 = remove_image_type_from_4D(final_array_4, image_type)

                
                if image_type == "brightfield":
                    assert np.array_equal(final_array_1[0], new_array_1[0])
                    assert np.array_equal(final_array_2[0], new_array_2[0])
                    assert np.array_equal(final_array_3[0], new_array_3[0])
                    assert np.array_equal(final_array_4[0], new_array_4[0])

                    assert np.array_equal(final_array_1[1], new_array_1[1])
                    assert np.array_equal(final_array_2[1], new_array_2[1])
                    assert np.array_equal(final_array_3[1], new_array_3[1])
                    assert np.array_equal(final_array_4[1], new_array_4[1])

                elif image_type == "reflective":
                    assert np.array_equal(final_array_1[0], new_array_1[0])
                    assert np.array_equal(final_array_2[0], new_array_2[0])
                    assert np.array_equal(final_array_3[0], new_array_3[0])
                    assert np.array_equal(final_array_4[0], new_array_4[0])

                    assert np.array_equal(final_array_1[2], new_array_1[1])
                    assert np.array_equal(final_array_2[2], new_array_2[1])
                    assert np.array_equal(final_array_3[2], new_array_3[1])
                    assert np.array_equal(final_array_4[2], new_array_4[1])
                elif image_type == "DAPI":
                    assert np.array_equal(final_array_1[1], new_array_1[0])
                    assert np.array_equal(final_array_2[1], new_array_2[0])
                    assert np.array_equal(final_array_3[1], new_array_3[0])
                    assert np.array_equal(final_array_4[1], new_array_4[0])

                    assert np.array_equal(final_array_1[2], new_array_1[1])
                    assert np.array_equal(final_array_2[2], new_array_2[1])
                    assert np.array_equal(final_array_3[2], new_array_3[1])
                    assert np.array_equal(final_array_4[2], new_array_4[1])
                else:
                    raise ValueError("Invalid image type")
                
                final_path = os.path.join(target_directory, directory, dir)
                checkDirectory(final_path)
                np.save(os.path.join(final_path, r"Final_5D_array_1.npy"), new_array_1)
                np.save(os.path.join(final_path, r"Final_5D_array_2.npy"), new_array_2)
                np.save(os.path.join(final_path, r"Final_5D_array_3.npy"), new_array_3)
                np.save(os.path.join(final_path, r"Final_5D_array_4.npy"), new_array_4)
                np.save(os.path.join(final_path, r"Label_matrix.npy"), label_array)
                

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\step9 Rotated_ROI_without_1600"
    target_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\No_brightfield_without_1600"
    cut_single_image_type(starting_directory, target_directory, "brightfield")




'''
         for file in files:
            x = str(file)
            if x.endswith(".txt"):
                file1 = open(os.path.join(starting_directory, directory, file))
                if test == True:
                    print(file1.read())
                    print(os.path.join(starting_directory, directory, file))
                    print(os.path.join(target_directory, directory, file))
                shutil.copy(os.path.join(starting_directory, directory, file),os.path.join(target_directory, directory, file))
'''
