'''
Code to find every cell tif (not any of the other processing tifs) inside a folder and just count them,
splitting the count into fibroblasts and cancer cell counts.

Glob path: just put in path instead of path.name


'''


import shutil
import glob
from pathlib import Path
import os
import re


# The directory to start searching in recursively.
starting_directory = r'D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X (no standard) Images Macro V5\step1 TIFFs and PNGs'


#for path in Path(r'C:\Users\Nicho\Dropbox\MIT Tumor Identification Project\AI project Tuan\Batch 1\Good Stuff').rglob('*.tif'):
fibroblast_counter = 0
cancer_cell_counter = 0
print("Printing all the accuracy names")
for path in Path(starting_directory).rglob('*.tif'):
    if "accuracy" in path.name:
        print(path.name)
        x = str(path.name)
        for ii in range(len(x)):
            if x[ii] == "F":
                if x[ii+1] == "b":
                    letters_before_fb = x[:ii]
                    letters_after_fb = x[ii:]
                    print("letters before fb = " + str(letters_before_fb))
                    print("letters after fb = " + str(letters_after_fb))
                    list_of_numbers_before_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_before_fb)]
                    list_of_numbers_after_fb = [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_fb)]
                    #print("numbers before fb = " + str(list_of_numbers_before_fb))
                    #print("numbers after fb = " + str(list_of_numbers_after_fb))
                    fibroblast_number = list_of_numbers_before_fb[-1]
                    cancer_cell_number = list_of_numbers_after_fb[0]
                    print("fibroblast number = " + str(fibroblast_number))
                    print("cancer cell number = " + str(list_of_numbers_after_fb[0]))
                    if fibroblast_number > cancer_cell_number:
                        fibroblast_counter = fibroblast_counter + 1
                    else:
                        cancer_cell_counter = cancer_cell_counter + 1

print("Total number of fibroblasts is " + str(fibroblast_counter))
print("Total number of cancer cells is " + str(cancer_cell_counter))



# This works, but sometimes there are values of NaN which screw up the position. Must find something more reliable.
'''
list_of_numbers = ([float(s) for s in re.findall(r'-?\d+\.?\d*',x)])
print(list_of_numbers)
print("Accuracy = " + str(list_of_numbers[6]))
'''
# res = [int(i) for i in x.split() if i.isdigit()]
#print("array of numbers in the path name = " + str(res))
#shutil.copy(str(path),r"C:\Users\Nicho\Desktop\Test results")




