'''
8_28_20
Program to search and move directories based on the previous "get_all_strange_values"

'''
import os
import shutil

'''
Function to move the strange cells (cells outside our normal boundaries) to a new folder.
Parameters
----------
starting_directory : string
    The directory to search for the strange cells in.
destination_directory : string
    The directory to move the strange cells to.
strange_list : list
    A list of strings of the strange cells to move. The function will search them out in the starting_directory.
Returns
-------
None.

'''
def search_and_move_strange_cells(starting_directory, destination_directory, strange_list):
    for value in strange_list:
        the_path = os.path.join(starting_directory,value)
        new_path = os.path.join(destination_directory,value)
        try:
            shutil.move(the_path,new_path)
        except FileNotFoundError as e:
            print("FileNotFoundError: %s : %s" % (the_path, e.strerror))


if __name__ == "__main__":
    starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\step3 2D image standard"
    destination_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X tile_watch\Large Cells"
    strange_list = ['device 2 ROI10_01.oib - Series 1-1 0001_cell3_0.2448Fb0.7552Tc_accuracy24.8719index57', 'device 2 ROI4_01.oib - Series 1 0010_cell2_0.5758Fb0.4242Tc_accuracy12.5566index45', 'device 2 ROI5_01.oib - Series 1-1 0004_cell15_0.4171Fb0.5829Tc_accuracy37.8183index52', 'device 2 ROI6_01.oib - Series 1-1 0004_cell15_0.4189Fb0.5811Tc_accuracy26.374index60', 'device 2 ROI6_01.oib - Series 1-1 0004_cell22_0.5062Fb0.4938Tc_accuracy36.8691index83', 'device 3 chip 1 2 3 ROI4_01-1 0010_cell4_0.3842Fb0.6158Tc_accuracy39.1028index41', 'device 3 chip 1 2 3 ROI5_01-1 0000_cell1_0.1704Fb0.8296Tc_accuracy30.6374index46', 'device 3 chip 1 2 3 ROI5_01-1 0006_cell2_0.06704Fb0.933Tc_accuracy59.9185index81', 'device 3 chip 3 ROI5_01-1 0008_cell4_0.6052Fb0.3948Tc_accuracy49.0022index40', 'device 3 chip 3 ROI7_01-1 0002_cell18_0.3552Fb0.6448Tc_accuracy14.5086index80', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0004_cell9_0.4481Fb0.5519Tc_accuracy10.6192index46', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0006_cell4_0.4796Fb0.5204Tc_accuracy18.4755index17', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0009_cell6_0.06346Fb0.9365Tc_accuracy48.5398index38', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0009_cell9_1Fb0Tc_accuracy27.8823index22', 'device 1 chip 1 and 2 ROI3_02.oib - Series 1-1 0001_cell9_1Fb0Tc_accuracy43.89index20', 'device 1 chip 1 and 2 ROI4_02.oib - Series 1-1 0005_cell9_0.5251Fb0.4749Tc_accuracy40.6135index56', 'device 1 chip 1 and 2 ROI5_02.oib - Series 1-1 0002_cell1_0.7283Fb0.2717Tc_accuracy50.3056index15', 'device 1 chip 1 and 2 ROI5_02.oib - Series 1-1 0006_cell17_0.1995Fb0.8005Tc_accuracy28.9508index82', 'device 1 chip 1 and 2 ROI6_02.oib - Series 1-1 0001_cell1_0.1734Fb0.8266Tc_accuracy56.6716index23', 'device 1 chip 1 and 2 ROI6_02.oib - Series 1-1 0005_cell0_0.6257Fb0.3743Tc_accuracy42.0865index35', 'device 1 chip 1 and 2 ROI7_02.oib - Series 1-1 0006_cell4_0.466Fb0.534Tc_accuracy57.6292index26', 'device 1 chip 1 and 2 ROI7_02.oib - Series 1-1 0006_cell8_0.73Fb0.27Tc_accuracy17.0021index34', 'device 1 chip 1 and 2 ROI8_02.oib - Series 1-1 0006_cell2_0.5905Fb0.4095Tc_accuracy4.1954index38', 'device 1 chip 3 ROI1_01.oib - Series 1-1 0005_cell3_0.2854Fb0.7146Tc_accuracy54.2903index49', 'device 1 chip 3 ROI1_01.oib - Series 1-1 0010_cell0_0.7048Fb0.2952Tc_accuracy20.1388index27', 'device 1 chip 3 ROI2_01.oib - Series 1-1 0006_cell1_0.2745Fb0.7255Tc_accuracy74.9484index5', 'device 1 chip 3 ROI3_01.oib - Series 1-1 0006_cell12_0.8906Fb0.1094Tc_accuracy34.2033index43', 'device 1 chip 3 ROI3_01.oib - Series 1-1 0010_cell10_0.3351Fb0.6649Tc_accuracy70.0556index62', 'device 1 chip 3 ROI3_01.oib - Series 1-1 0010_cell5_0.5484Fb0.4516Tc_accuracy36.921index36', 'device 1 chip 3 ROI5_01-1 0009_cell2_0.3287Fb0.6713Tc_accuracy22.9167index47', 'device 1 chip 3 ROI5_01-1 0010_cell1_0.2978Fb0.7022Tc_accuracy14.611index32', 'device 1 chip 3 ROI6_01-1 0014_cell5_0.359Fb0.641Tc_accuracy34.6082index47', 'device 1 chip 3 ROI7_01-1 0013_cell1_0.3687Fb0.6313Tc_accuracy13.6766index59', 'device 1 chip 3 ROI7_01-1 0014_cell1_0.6425Fb0.3575Tc_accuracy41.4251index23', 'device 1 chip 3 ROI7_01-1 0014_cell6_0.3166Fb0.6834Tc_accuracy59.7642index58', 'device 1 chip 3 ROI7_01-1 0014_cell8_0.5274Fb0.4726Tc_accuracy37.4709index23', 'device 1 chip 3 ROI8_01-1 0013_cell1_0.4614Fb0.5386Tc_accuracy21.1473index54', 'device 2 ROI10_01.oib - Series 1-1 0001_cell6_0.4472Fb0.5528Tc_accuracy22.2411index75', 'device 2 ROI1_01.oib - Series 1-1 0006_cell6_0.7868Fb0.2132Tc_accuracy86.0257index24', 'device 2 ROI2_01.oib - Series 1-1 0005_cell9_0.5123Fb0.4877Tc_accuracy97.4288index22', 'device 2 ROI4_01.oib - Series 1 0005_cell4_0.5208Fb0.4792Tc_accuracy29.5761index72', 'device 2 ROI4_01.oib - Series 1 0005_cell9_0.5085Fb0.4915Tc_accuracy23.8466index108', 'device 2 ROI4_01.oib - Series 1 0010_cell13_0.6167Fb0.3833Tc_accuracy50.4028index66', 'device 2 ROI4_01.oib - Series 1 0010_cell7_0.2311Fb0.7689Tc_accuracy31.886index94', 'device 2 ROI5_01.oib - Series 1-1 0000_cell1_0.3189Fb0.6811Tc_accuracy34.8876index13', 'device 2 ROI5_01.oib - Series 1-1 0001_cell14_0.5733Fb0.4267Tc_accuracy40.3544index98', 'device 2 ROI7_01.oib - Series 1-1 0006_cell13_0.8888Fb0.1112Tc_accuracy89.9686index24', 'device 2 ROI9_01.oib - Series 1-1 0008_cell11_0.6228Fb0.3772Tc_accuracy21.4417index43', 'device 2 ROI9_01.oib - Series 1-1 0009_cell26_0.5351Fb0.4649Tc_accuracy43.0967index82', 'device 3 chip 1 2 3 ROI4_01-1 0015_cell9_0.9854Fb0.01457Tc_accuracy66.1846index37', 'device 3 chip 1 2 3 ROI5_01-1 0005_cell8_0.6843Fb0.3157Tc_accuracy32.3698index38', 'device 3 chip 1 2 3 ROI5_01-1 0006_cell32_0.6691Fb0.3309Tc_accuracy27.5455index72', 'device 3 chip 1 2 3 ROI5_01-1 0011_cell7_0.5514Fb0.4486Tc_accuracy49.1776index57', 'device 3 chip 1 2 3 ROI5_01-1 0016_cell3_0.6919Fb0.3081Tc_accuracy41.7958index62', 'device 3 chip 1 2 3 ROI5_01-1 0022_cell3_0.9712Fb0.02877Tc_accuracy81.725index13', 'device 3 chip 1 2 3 ROI6_01-1 0011_cell20_0.659Fb0.341Tc_accuracy70.8364index47', 'device 3 chip 1 2 3 ROI8_01-1 0001_cell1_0.8918Fb0.1082Tc_accuracy9.4124index4', 'device 3 chip 3 ROI2_01-1 0009_cell16_0.3856Fb0.6144Tc_accuracy83.8748index73', 'device 3 chip 3 ROI2_01-1 0014_cell7_1Fb0Tc_accuracy69.0661index25', 'device 3 chip 3 ROI7_01-1 0002_cell24_0.4319Fb0.5681Tc_accuracy65.9294index55', 'device 3 chip 3 ROI7_01-1 0003_cell16_0.819Fb0.181Tc_accuracy31.0345index35', 'device 3 chip 3 ROI7_01-1 0018_cell5_0.476Fb0.524Tc_accuracy25.8014index54', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0004_cell14_0.8862Fb0.1138Tc_accuracy79.36index35', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0009_cell8_0.6289Fb0.3711Tc_accuracy44.0973index18', 'device 1 chip 1 and 2 ROI2_02.oib - Series 1-1 (cropped) 0010_cell2_0.01331Fb0.9867Tc_accuracy21.7228index27', 'device 1 chip 1 and 2 ROI4_02.oib - Series 1-1 0005_cell24_0.625Fb0.375Tc_accuracy67.0418index61', 'device 1 chip 1 and 2 ROI5_02.oib - Series 1-1 0005_cell9_0.421Fb0.579Tc_accuracy76.8077index32', 'device 1 chip 1 and 2 ROI5_02.oib - Series 1-1 0006_cell12_0.6336Fb0.3664Tc_accuracy26.5486index43', 'device 1 chip 1 and 2 ROI5_02.oib - Series 1-1 0006_cell18_0.4691Fb0.5309Tc_accuracy28.1003index90', 'device 1 chip 1 and 2 ROI9_02.oib - Series 1-1 0009_cell0_0.5004Fb0.4996Tc_accuracy62.6888index3', 'device 1 chip 3 ROI1_01.oib - Series 1-1 0010_cell4_0.839Fb0.161Tc_accuracy21.2232index21', 'device 1 chip 3 ROI2_01.oib - Series 1-1 0005_cell5_0Fb1Tc_accuracy63.2804index45', 'device 1 chip 3 ROI3_01.oib - Series 1-1 0006_cell7_0.1095Fb0.8905Tc_accuracy32.6474index73', 'device 1 chip 3 ROI3_01.oib - Series 1-1 0010_cell13_0.3205Fb0.6795Tc_accuracy40.34index67', 'device 1 chip 3 ROI5_01-1 0004_cell1_0.6478Fb0.3522Tc_accuracy64.7901index10', 'device 1 chip 3 ROI5_01-1 0009_cell3_0.5903Fb0.4097Tc_accuracy30.8436index24', 'device 1 chip 3 ROI7_01-1 0008_cell0_0.5804Fb0.4196Tc_accuracy33.3865index18', 'device 1 chip 3 ROI7_01-1 0013_cell6_0.09128Fb0.9087Tc_accuracy33.1091index64', 'device 1 chip 3 ROI8_01-1 0022_cell6_0.6562Fb0.3438Tc_accuracy35.3213index12', 'device 2 ROI10_01.oib - Series 1-1 0000_cell1_0.05497Fb0.945Tc_accuracy68.3305index31', 'device 2 ROI1_01.oib - Series 1-1 0001_cell7_1Fb0Tc_accuracy33.0102index25', 'device 2 ROI2_01.oib - Series 1-1 0006_cell4_0.8782Fb0.1218Tc_accuracy45.2775index20', 'device 2 ROI4_01.oib - Series 1 0005_cell29_0.2928Fb0.7072Tc_accuracy40.4593index133', 'device 2 ROI5_01.oib - Series 1-1 0004_cell22_0.1804Fb0.8196Tc_accuracy41.2883index105', 'device 2 ROI6_01.oib - Series 1-1 0004_cell33_0.4476Fb0.5524Tc_accuracy33.8451index85', 'device 2 ROI6_01.oib - Series 1-1 0004_cell42_0.334Fb0.666Tc_accuracy31.5758index135', 'device 2 ROI6_01.oib - Series 1-1 0008_cell4_0.7118Fb0.2882Tc_accuracy79.9411index18', 'device 2 ROI8_01.oib - Series 1-1 0002_cell4_0.6225Fb0.3775Tc_accuracy71.4676index17', 'device 2 ROI9_01.oib - Series 1-1 0009_cell18_0.3857Fb0.6143Tc_accuracy40.0312index116', 'device 3 chip 1 2 3 ROI1_01-1 0006_cell7_0.3202Fb0.6798Tc_accuracy29.6316index34', 'device 3 chip 1 2 3 ROI1_01-1 0011_cell14_0.7897Fb0.2103Tc_accuracy74.6831index27', 'device 3 chip 1 2 3 ROI2_01-1 0016_cell0_0.3931Fb0.6069Tc_accuracy36.3907index7', 'device 3 chip 1 2 3 ROI2_01-1 0018_cell6_0.4807Fb0.5193Tc_accuracy71.5367index19', 'device 3 chip 1 2 3 ROI4_01-1 0011_cell12_0.686Fb0.314Tc_accuracy95.3632index39', 'device 3 chip 1 2 3 ROI4_01-1 0016_cell9_0.6788Fb0.3212Tc_accuracy98.3319index31', 'device 3 chip 1 2 3 ROI5_01-1 0010_cell8_0.839Fb0.161Tc_accuracy79.7829index30', 'device 3 chip 3 ROI2_01-1 0014_cell5_1Fb0Tc_accuracy19.3696index23', 'device 3 chip 3 ROI5_01-1 0008_cell9_0.2025Fb0.7975Tc_accuracy80.9809index88', 'device 3 chip 3 ROI7_01-1 0013_cell15_0.2248Fb0.7752Tc_accuracy91.8513index49']
    search_and_move_strange_cells(starting_directory, destination_directory, strange_list)
