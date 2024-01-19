'''
8/28/20

Our team decided on a 20-50-50 standard based on the results of this program of the get_standard_for_padding.py program.

There turned out to be a small set of cells with abnormal dimensions outside of this standard.
Upon taking a look at these cells, the vast majority of them appeared to be improperly cropped by the segmentation algorithm. The images did not resemble cells and had bits of other cells or the background in them.

I nicknamed them "strange cells" because of their appearance and the fact that they were outside of the image standard.

This program is designed to find all the cells with dimensions outside of the 20-50-50 standard, and return their directories so that they can be removed from the dataset.

TODO: Deduplication won't be necessary if you just use a combined list from the start. Since we're no longer inspecting the dictionaries individually, there's probably not a need to seperate out the dictionaries.
'''




import os
from Common_Utils.image_os_walker import image_os_walker


'''
Determine if the dimensions of a 3D matrix are outside of the image standard, and if so, store its directory inside a dictionary to keep track of it.
Parameters
----------
the_matrix: numpy matrix
    The 3D matrix to get the dimensions of.
dir: string
    The directory the 3D matrix is located in.
strange_z_values_dictionary: dictionary
    The dictionary to store the directories of the 3D matrices with strange z values in.
strange_y_values_dictionary: dictionary 
    The dictionary to store the directories of the 3D matrices with strange y values in.
strange_x_values_dictionary: dictionary
    The dictionary to store the directories of the 3D matrices with strange x values in.
image_standard: list
    The standard to compare the dimensions of the 3D matrices to. image_standard[0] is the standard for z, image_standard[1] is the standard for y, and image_standard[2] is the standard for x. Default is [20,50,50].
Returns
-------
strange_z_values_dictionary: dictionary
    The dictionary to store the directories of the 3D matrices with strange z values in.
strange_y_values_dictionary: dictionary
    The dictionary to store the directories of the 3D matrices with strange y values in.
strange_x_values_dictionary: dictionary
    The dictionary to store the directories of the 3D matrices with strange x values in.

'''
def find_strange_values_in_dir(the_matrix, dir, strange_z_values_dictionary, strange_y_values_dictionary, strange_x_values_dictionary, image_standard = [20,50,50]):
    max_z = len(the_matrix)
    max_y = 0
    max_x = 0
    for z in range(len(the_matrix)):
        if len(the_matrix[z]) > max_y:
            max_y = len(the_matrix[z])
        for y in range(len(the_matrix[z])):
            if len(the_matrix[z][y]) > max_x:
                max_x = len(the_matrix[z][y])
    if max_z > image_standard[0]:
        strange_z_values_dictionary[dir] = max_z
    if max_y > image_standard[1]:
        strange_y_values_dictionary[dir] = max_y
    if max_x > image_standard[2]:
        strange_x_values_dictionary[dir] = max_x
    return strange_z_values_dictionary, strange_y_values_dictionary, strange_x_values_dictionary

'''
Searches out all the cells with dimensions outside of the image standard in starting_directory, and returns their directories in a list so that they can be removed from the dataset.
Parameters
----------
starting_directory : string
    The directory to start searching in recursively.
image_standard: list
    The standard to compare the dimensions of the 3D matrices to. image_standard[0] is the standard for z, image_standard[1] is the standard for y, and image_standard[2] is the standard for x. Default is [20,50,50].
test : boolean
    Whether or not to print out extra information for testing purposes.
Returns
-------
combined_list: list
    The list of directories of the cells with dimensions outside of the image standard. 
'''
def get_all_strange_values(starting_directory, image_standard=[20,50,50], test = False):
    # The 3 3D arrays off this data
    DAPI_3D_array = []
    Reflection_3D_array = []
    Transmission_brightfield_3D_array = []

    strange_x_values_dictionary = {}
    strange_y_values_dictionary = {}
    strange_z_values_dictionary = {}

    for root, dirs, files in os.walk(starting_directory):
        for dir in dirs:
            # Begin searching each directory for the 2D PNGs
            print("Now searching directory " + str(dir))
            image_os_walker(starting_directory, dir, DAPI_3D_array, Reflection_3D_array, Transmission_brightfield_3D_array, test)
            # Padding the matrices
            find_strange_values_in_dir(DAPI_3D_array, dir, strange_z_values_dictionary, strange_y_values_dictionary, strange_x_values_dictionary, image_standard)
            find_strange_values_in_dir(Reflection_3D_array, dir, strange_z_values_dictionary, strange_y_values_dictionary, strange_x_values_dictionary, image_standard)
            find_strange_values_in_dir(Transmission_brightfield_3D_array, dir, strange_z_values_dictionary, strange_y_values_dictionary, strange_x_values_dictionary, image_standard)

            # reset the 3D matrices after saving them.
            DAPI_3D_array = []
            Reflection_3D_array = []
            Transmission_brightfield_3D_array = []
    if test == True:
        print("The strange values for x are " + str(strange_x_values_dictionary))
        print("The strange values for y are " + str(strange_y_values_dictionary))
        print("The strange values for z are " + str(strange_z_values_dictionary))


    # De-duplication and combining the strange value lists together so that they can be removed together.
    z_list = []
    for value in strange_z_values_dictionary:
        z_list.append(value)
    y_list = []
    for value in strange_y_values_dictionary:
        y_list.append(value)
    x_list = []
    for value in strange_x_values_dictionary:
        x_list.append(value)
    combined_list = []
    for z in z_list:
        combined_list.append(z)
    duplicate = False
    for y in y_list:
        for z in z_list:
            if z == y:
                duplicate = True
        if duplicate == False:
            combined_list.append(y)
        duplicate = False
    duplicate = False
    for x in x_list:
        for z in z_list:
            if x == z:
                duplicate = True
        for y in y_list:
            if x == y:
                duplicate = True
        if duplicate == False:
            combined_list.append(x)
        duplicate = False

    # Tally up the number of strange values.
    if test == True:
        print(combined_list)
        for value in combined_list:
            print(value)
        print(len(combined_list))
    
    return combined_list

if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step3 2D image standard"
    get_all_strange_values(starting_directory)
