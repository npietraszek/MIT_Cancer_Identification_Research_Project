'''
Slight modification of the usual cell counter. Instead of counting the cells when they are in .tif form, this counts the number of cells
when they are represented as numpy matrices instead by looking at the information in their folder names.
'''



import os
import re

'''
Simple function to count the number of fibroblasts and cancer cells in a folder.
Parameters
----------
starting_directory : str
    The directory to start searching in recursively.    
Returns
-------
fibroblast_counter : int
    The number of fibroblasts in the folder.
cancer_cell_counter : int
    The number of cancer cells in the folder.

'''
def matrix_cell_counter(starting_directory, test=False):
    # The directory to start searching in recursively.
    fibroblast_counter = 0
    cancer_cell_counter = 0
    if test == True:
        print("Printing all the accuracy names")
    for root, dirs, files in os.walk(starting_directory):
        for the_dir in dirs:
            if "accuracy" in the_dir:
                if test == True:
                    print(the_dir)
                x = str(the_dir)
                for ii in range(len(x)):
                    if x[ii] == "F":
                        if x[ii+1] == "b":
                            letters_before_fb = x[:ii]
                            letters_after_fb = x[ii:]
                            if test == True:
                                print("letters before fb = " + str(letters_before_fb))
                                print("letters after fb = " + str(letters_after_fb))
                            list_of_numbers_before_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_before_fb)]
                            list_of_numbers_after_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_fb)]
                            fibroblast_number = list_of_numbers_before_fb[-1]
                            cancer_cell_number = list_of_numbers_after_fb[0]
                            if test == True:
                                print("fibroblast number = " + str(fibroblast_number))
                                print("cancer cell number = " + str(list_of_numbers_after_fb[0]))
                            if fibroblast_number > cancer_cell_number:
                                fibroblast_counter = fibroblast_counter + 1
                            else:
                                cancer_cell_counter = cancer_cell_counter + 1
    if test == True:
        print("Total number of fibroblasts is " + str(fibroblast_counter))
        print("Total number of cancer cells is " + str(cancer_cell_counter))
    return (fibroblast_counter,cancer_cell_counter)

if __name__ == "__main__":
    starting_directory = r'D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X 90confidence groundcell\step7 Rotated 4D matrices (20-50-50)'
    matrix_cell_counter(starting_directory)