'''
8/29/20

Program to remove excess cancer cells by removing the ones with the lowest confidences in classifcation.



'''




import os
import shutil
import re
from Common_Utils.checkDirectory import checkDirectory
from Common_Utils.find_accuracy_number import find_accuracy_number
def harsh_balancer(starting_directory, new_directory, moving_directory, num_of_cancer_cells_to_remove, test=False):
    checkDirectory(starting_directory)
    checkDirectory(new_directory)
    checkDirectory(moving_directory)
    # We need to remove cancer cells
    cancer_cell_counter = 0
    fibroblast_counter = 0
    list_of_cancer_cells = []
    list_of_accuracy_values = []
    list_of_confidences = []


    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            x = str(dir)

            # Checking the probability values for the cell classifcation.
            for ii in range(len(x)):
                if x[ii] == "F":
                    if x[ii + 1] == "b":
                        letters_before_fb = x[:ii]
                        letters_after_fb = x[ii:]
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
                            if fibroblast_number >= 0.5:
                                the_path = os.path.join(starting_directory, dir)
                                destination = os.path.join(new_directory, dir)
                                shutil.copytree(the_path, destination)
                                fibroblast_counter = fibroblast_counter + 1
                        else:
                            if cancer_cell_number >= 0.5:
                                # Check whether or not the accuracy value is above 90%.
                                accuracy_value = find_accuracy_number(x)
                                list_of_cancer_cells.append(dir)
                                list_of_confidences.append(cancer_cell_number)
                                list_of_accuracy_values.append(accuracy_value)
                                cancer_cell_counter = cancer_cell_counter + 1

    for x in range(num_of_cancer_cells_to_remove):
        current_minimum = min(list_of_confidences)
        for i in range(len(list_of_cancer_cells)):
            if list_of_confidences[i] == current_minimum:
                # A minimum confidence has been found. Pop the values there...
                list_of_accuracy_values.pop(i)

                # Move this cancer cell elsewhere...
                print(list_of_cancer_cells[i])
                the_path = os.path.join(starting_directory, list_of_cancer_cells[i])
                destination = os.path.join(moving_directory, list_of_cancer_cells[i])
                shutil.copytree(the_path, destination)
                list_of_cancer_cells.pop(i)
                list_of_confidences.pop(i)
                cancer_cell_counter = cancer_cell_counter - 1
                break
    if test == True:
        print("Number of cancer cells before curation = " + str(cancer_cell_counter))
        print("Number of fibroblast cells before curation = " + str(fibroblast_counter))
    # After the weakest cancer cells have been removed, continue
    for x in list_of_cancer_cells:
        print("You pass! " + str(x) + " passes the test!")
        the_path = os.path.join(starting_directory, x)
        destination = os.path.join(new_directory, x)
        shutil.copytree(the_path, destination)
    if test == True:
        print("Curation has finished.")
        print("Number of cancer cells after curation = " + str(cancer_cell_counter))
        print("Number of fibroblast cells after curation = " + str(fibroblast_counter))

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step5.5 curated 4D matrices (20-50-50)"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step6 Balanced 4D matrices"
    moving_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\unused_cancer_cells"
    harsh_balancer(starting_directory, new_directory, moving_directory, 1879)