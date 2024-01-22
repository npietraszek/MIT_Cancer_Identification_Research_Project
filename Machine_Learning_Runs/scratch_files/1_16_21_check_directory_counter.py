
#import os

#files = folders = 0

#for _, dirnames, filenames in os.walk(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step4 3D matrices"):
#  # ^ this idiom means "we won't be using this value"
#    files += len(filenames)
#    folders += len(dirnames)

#print ("{:,} files, {:,} folders".format(files, folders))

#import os

#print("Other method")
#print(len(next(os.walk(r'D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step4 3D matrices'))[1]))

import os

dir = r"device 1 chip 1 and 2 ROI1_02.oib - Series 1-1 0000_cell1_0.7695Fb0.2305Tc_accuracy75.5408index6"

path_to_walk = os.path.join(
  r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step7 Rotated 4D matrices",dir)
print(str(path_to_walk))
print(len(next(os.walk(path_to_walk))[
            1]))