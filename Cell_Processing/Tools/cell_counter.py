'''
Code to find every cell tif (not any of the other processing tifs) inside a folder and just count them,
splitting the count into fibroblasts and cancer cell counts.

Glob path: just put in path instead of path.name


'''


from pathlib import Path
import re


'''
Simple function to count the number of fibroblasts and cancer cells in a folder.
Parameters
----------
starting_directory : str
    The directory to start searching in recursively.
test : bool
    Whether or not to print out the names of the accuracy files.
Returns
-------
fibroblast_counter : int
    The number of fibroblasts in the folder.
cancer_cell_counter : int
    The number of cancer cells in the folder.
'''
def cell_counter(starting_directory, test=False):
    # The directory to start searching in recursively.
    fibroblast_counter = 0
    cancer_cell_counter = 0
    if test == True:
        print("Printing all the accuracy names")
    for path in Path(starting_directory).rglob('*.tif'):
        if "accuracy" in path.name:
            if test == True:
                print(path.name)
            x = str(path.name)
            for ii in range(len(x)):
                if x[ii] == "F":
                    if x[ii+1] == "b":
                        # Find the fibroblast confidence and the cancer cell confidence and compare them.
                        # Whichever is higher is the classification of the cell.
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

    print("Total number of fibroblasts is " + str(fibroblast_counter))
    print("Total number of cancer cells is " + str(cancer_cell_counter))
    return (fibroblast_counter,cancer_cell_counter)

if __name__ == "__main__":
    starting_directory = r'D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X (no standard) Images Macro V5\step1 TIFFs and PNGs'
    cell_counter(starting_directory)



