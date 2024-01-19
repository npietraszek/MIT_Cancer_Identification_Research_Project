'''
8/10/20
Code designed to find the cells that pass each decile of accuracy percentages for diagnostic purposes.
'''




import os
from pathlib import Path
import re
from Common_Utils.checkDirectory import checkDirectory


'''
Code designed to find the cells that pass each decile of accuracy percentages for diagnostic purposes.
Parameters
----------
starting_directory : str
    The directory to start searching in recursively.
new_directory : str
    The directory to copy the cells that pass the accuracy test to.
Returns
-------
None
'''
def multifile_cell_searcher(starting_directory, new_directory):
    for the_loop_counter in range(0,10):
        target_directory = new_directory + r"\{0}".format(the_loop_counter) + r"0% decile check"
        
        checkDirectory(starting_directory)
        checkDirectory(new_directory)
        checkDirectory(target_directory)

        fibroblast_counter = 0
        cancer_cell_counter = 0
        total_cell_count = 0
        for the_file in Path(starting_directory).rglob('*.tif'):
            x = str(the_file)

            accuracy_test = False
            classification_test = False
            is_cell_fibroblast = False
            is_cell_cancer_cell = False

            # Check whether or not the accuracy value is above 90%.
            accuracy_index = x.find("accuracy")
            if accuracy_index != -1:
                letters_after_accuracy = x[accuracy_index+8:]
                total_cell_count = total_cell_count + 1
                # letters_before_accuracy = x[:ii+7]
                letters_after_accuracy = x[ii + 8:]
                print(letters_after_accuracy)
                list_of_numbers = (
                [float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_accuracy)])
                print(list_of_numbers)
                accuracy_value = list_of_numbers[0]
                float(10*the_loop_counter)
                if accuracy_value > float(10*the_loop_counter) and accuracy_value <= float(10*(the_loop_counter+1)):
                    accuracy_test = True
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
                                classification_test = True
                                is_cell_fibroblast = True
                        else:
                            if cancer_cell_number >= 0.5:
                                classification_test = True
                                is_cell_cancer_cell = True
            if classification_test == True:
                if accuracy_test == True:
                    print("You pass! " + str(x) + " passes the test!")
                    the_path = os.path.join(starting_directory,the_file)
                    destination = target_directory
                    #shutil.copy(the_path,destination)
                    if is_cell_fibroblast:
                        fibroblast_counter = fibroblast_counter + 1
                    if is_cell_cancer_cell:
                        cancer_cell_counter = cancer_cell_counter + 1

        print("Curation has finished.")
        print("Number of cancer cells = " + str(cancer_cell_counter))
        print("Number of fibroblast cells = " + str(fibroblast_counter))
        print("Total number of cells = " + str(total_cell_count))
        file1 = open(os.path.join(new_directory, r"10_29_20_test_results_{0}".format(the_loop_counter) + "0% - {0}0%".format(the_loop_counter + 1)  + "_decile.txt"), "a")  # append mode
        file1.write("10_29_20_test_results_{0}".format(the_loop_counter) + "0% - {0}0%".format(the_loop_counter + 1)  + "_decile" + "\n")
        file1.write("Number of cancer cells = " + str(cancer_cell_counter) + "\n")
        file1.write("Number of fibroblast cells = " + str(fibroblast_counter) + "\n")
        file1.write("Number of cells under current decile = " + str(cancer_cell_counter+fibroblast_counter) + "\n")
        file1.write("Total number of cells = " + str(total_cell_count) + "\n")
        file1.write("       ----END---- \n \n \n")
        file1.close()


if __name__ == '__main__':
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\NEW 20X cell images\new 20x batch Testing with Tuan's V3 Macro"
    new_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\20X images Decile Checks"
    multifile_cell_searcher(starting_directory, new_directory)