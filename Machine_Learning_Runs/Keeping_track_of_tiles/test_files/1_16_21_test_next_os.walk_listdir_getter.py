import os

saved_1600_cells_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_1600_cells"
print(str(saved_1600_cells_directory))




list_of_directories_to_walk = next(os.walk(saved_1600_cells_directory))[1]

saved_total_directory_list = []
for the_directory in list_of_directories_to_walk:
    full_directory = os.path.join(saved_1600_cells_directory,the_directory)
    for root, dirs, files in os.walk(full_directory):
        for dir in dirs:
            saved_total_directory_list.append(os.path.join(full_directory, dir))

print(len(saved_total_directory_list))


starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step9 Rotated_ROI_without_1600"
print(str(starting_directory))

list_of_directories_to_walk = next(os.walk(starting_directory))[1]

total_directory_list = []
for the_directory in list_of_directories_to_walk:
    full_directory = os.path.join(starting_directory,the_directory)
    for root, dirs, files in os.walk(full_directory):
        for dir in dirs:
            total_directory_list.append(os.path.join(full_directory, dir))

print(len(total_directory_list))


#print(next(os.walk(path_to_walk))[1])