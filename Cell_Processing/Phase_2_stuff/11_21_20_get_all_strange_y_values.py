'''
8/28/20

Program to discover and print out the list of cells with dimensions above the 20 30 30 standard we have set


Remaining strange values:
The strange values for x are {'ROI10_01.oib - Series 1-1 0000_cell1_0.1258Fb0.8742Tc_accuracy25.7104index23': 55, 'ROI10_01.oib - Series 1-1 0005_cell2_0Fb1Tc_accuracy30.411index30': 56, 'ROI1_01-1 0015_cell4_0Fb1Tc_accuracy32.6483index15': 145, 'ROI1_01-1 0018_cell8_0.7638Fb0.2362Tc_accuracy14.2019index38': 63, 'ROI1_01-1 0022_cell1_0.7639Fb0.2361Tc_accuracy39.4434index12': 145, 'ROI1_01-1 0023_cell11_0.3609Fb0.6391Tc_accuracy76.4256index33': 51, 'ROI1_01-1 0023_cell14_0.9944Fb0.005596Tc_accuracy74.4708index26': 62, 'ROI1_01-1 0025_cell9_1Fb0Tc_accuracy2.063index27': 51, 'ROI1_01-1 0032_cell15_0.7355Fb0.2645Tc_accuracy73.4977index42': 56, 'ROI1_01-1 0033_cell7_0.7588Fb0.2412Tc_accuracy6.2008index27': 63, 'ROI1_01.oib - Series 1-1 0001_cell5_1Fb0Tc_accuracy1.783index14': 55, 'ROI1_01.oib - Series 1-1 0002_cell7_1Fb0Tc_accuracy33.936index21': 55, 'ROI1_01.oib - Series 1-1 0006_cell12_1Fb0Tc_accuracy1.725index28': 55, 'ROI1_01.oib - Series 1-1 0007_cell14_1Fb0Tc_accuracy32.0918index32': 55, 'ROI1_01.oib - Series 1-1 0008_cell6_0.1423Fb0.8577Tc_accuracy9.4095index31': 51, 'ROI1_01.oib - Series 1-1 0013_cell3_0.1611Fb0.8389Tc_accuracy23.2223index34': 55, 'ROI1_01.oib - Series 1-1 0014_cell0_0.7105Fb0.2895Tc_accuracy17.0513index18': 82, 'ROI1_01.oib - Series 1-1 0014_cell2_0.9185Fb0.08148Tc_accuracy43.0786index33': 70, 'ROI1_01.oib - Series 1-1 0014_cell3_0.227Fb0.773Tc_accuracy23.9014index40': 57, 'ROI1_01.oib - Series 1-1 0015_cell3_0.9369Fb0.06313Tc_accuracy50.4062index23': 71, 'ROI1_01.oib - Series 1-1 0017_cell5_1Fb0Tc_accuracy15.1942index19': 61, 'ROI1_01.oib - Series 1-1 0018_cell9_0.77Fb0.23Tc_accuracy18.5052index24': 54, 'ROI1_01.oib - Series 1-1 0019_cell12_0.9126Fb0.08738Tc_accuracy1.2362index17': 88, 'ROI1_01.oib - Series 1-1 0020_cell2_0.9481Fb0.05195Tc_accuracy49.7759index31': 71, 'ROI1_01.oib - Series 1-1 0021_cell0_0.6892Fb0.3108Tc_accuracy14.5651index23': 83, 'ROI1_01.oib - Series 1-1 0021_cell4_0.7384Fb0.2616Tc_accuracy10.3712index24': 71, 'ROI1_01.oib - Series 1-1 0021_cell6_0.9155Fb0.08452Tc_accuracy20.3029index20': 63, 'ROI1_01.oib - Series 1-1 0022_cell2_1Fb0Tc_accuracy13.8428index13': 65, 'ROI1_02.oib - Series 0006_cell14_0.405Fb0.595Tc_accuracy40.5053index67': 53, 'ROI1_02.oib - Series 0007_cell17_0.4615Fb0.5385Tc_accuracy25.1084index53': 52, 'ROI1_02.oib - Series 0007_cell4_0.6625Fb0.3375Tc_accuracy21.6905index42': 79, 'ROI1_02.oib - Series 0007_cell7_0.6853Fb0.3147Tc_accuracy36.4385index45': 56, 'ROI1_02.oib - Series 0011_cell19_0.5422Fb0.4578Tc_accuracy24.8755index48': 52, 'ROI1_02.oib - Series 0012_cell16_0.4634Fb0.5366Tc_accuracy25.1668index49': 53, 'ROI1_02.oib - Series 0012_cell1_0.7733Fb0.2267Tc_accuracy10.0514index45': 57, 'ROI1_02.oib - Series 0012_cell6_0.6625Fb0.3375Tc_accuracy21.8217index40': 80, 'ROI1_02.oib - Series 0013_cell16_0.2189Fb0.7811Tc_accuracy4.2002index61': 68, 'ROI2_01-1 0007_cell2_0.9113Fb0.08872Tc_accuracy30.6453index48': 62, 'ROI2_01-1 0013_cell1_0.8003Fb0.1997Tc_accuracy8.1817index34': 64, 'ROI2_01-1 0018_cell4_1Fb0Tc_accuracy19.8389index13': 55, 'ROI2_01-1 0020_cell2_0.6511Fb0.3489Tc_accuracy4.03index38': 109, 'ROI2_01-1 0020_cell9_0.9354Fb0.06456Tc_accuracy8.1262index46': 70, 'ROI2_01-1 0025_cell0_1Fb0Tc_accuracy22.1956index8': 73, 'ROI2_01-1 0027_cell7_1Fb0Tc_accuracy19.3696index31': 55, 'ROI2_01-1 0031_cell9_0.726Fb0.274Tc_accuracy22.7933index35': 84, 'ROI2_01-1 0033_cell4_0Fb1Tc_accuracy28.9534index31': 58, 'ROI2_01-1 0034_cell3_1Fb0Tc_accuracy29.3983index20': 55, 'ROI2_01-1 0037_cell0_0.9424Fb0.05761Tc_accuracy23.5462index6': 73, 'ROI2_01-1 0040_cell6_0Fb1Tc_accuracy29.2544index19': 58, 'ROI2_01.oib - Series 1-1 0006_cell4_1Fb0Tc_accuracy77.0611index12': 71, 'ROI2_01.oib - Series 1-1 0008_cell3_0.9363Fb0.06366Tc_accuracy41.4544index21': 59, 'ROI2_01.oib - Series 1-1 0013_cell3_0.9779Fb0.02211Tc_accuracy41.7385index17': 58, 'ROI2_01.oib - Series 1-1 0013_cell5_0.2956Fb0.7044Tc_accuracy29.5626index57': 58, 'ROI2_01.oib - Series 1-1 0021_cell1_0Fb1Tc_accuracy40.3132index11': 70, 'ROI2_02.oib - Series 1-1 (cropped) 0002_cell7_0.3972Fb0.6028Tc_accuracy16.3362index16': 81, 'ROI2_02.oib - Series 1-1 (cropped) 0005_cell5_0.6162Fb0.3838Tc_accuracy6.3765index15': 82, 'ROI2_02.oib - Series 1-1 (cropped) 0006_cell14_0.4607Fb0.5393Tc_accuracy13.7984index44': 63, 'ROI2_02.oib - Series 1-1 (cropped) 0006_cell3_0.6163Fb0.3837Tc_accuracy4.3986index24': 74, 'ROI2_02.oib - Series 1-1 (cropped) 0011_cell5_0Fb1Tc_accuracy0.2411index25': 61, 'ROI2_02.oib - Series 1-1 (cropped) 0011_cell6_0.777Fb0.223Tc_accuracy44.6842index14': 57, 'ROI2_02.oib - Series 1-1 (cropped) 0013_cell3_0.8171Fb0.1829Tc_accuracy26.214index15': 90, 'ROI2_02.oib - Series 1-1 (cropped) 0014_cell1_1Fb0Tc_accuracy8.8602index6': 72, 'ROI2_02.oib - Series 1-1 (cropped) 0016_cell4_0.575Fb0.425Tc_accuracy5.2412index27': 60, 'ROI2_02.oib - Series 1-1 (cropped) 0016_cell9_0.7777Fb0.2223Tc_accuracy44.6842index17': 57, 'ROI3_01-1 0031_cell6_0.829Fb0.171Tc_accuracy13.4426index44': 55, 'ROI3_01-1 0043_cell0_1Fb0Tc_accuracy31.0317index8': 84, 'ROI3_01-1 0045_cell5_0Fb1Tc_accuracy4.1985index12': 57, 'ROI3_01.oib - Series 1-1 0009_cell5_0.1746Fb0.8254Tc_accuracy28.6016index51': 52, 'ROI3_01.oib - Series 1-1 0015_cell14_0.9994Fb0.0005548Tc_accuracy30.7207index41': 65, 'ROI3_01.oib - Series 1-1 0015_cell22_0.2276Fb0.7724Tc_accuracy15.52index75': 59, 'ROI3_01.oib - Series 1-1 0015_cell8_0.1794Fb0.8206Tc_accuracy27.0855index55': 52, 'ROI3_01.oib - Series 1-1 0016_cell11_0.6451Fb0.3549Tc_accuracy13.283index27': 77, 'ROI3_01.oib - Series 1-1 0016_cell12_0.5801Fb0.4199Tc_accuracy18.695index29': 59, 'ROI3_01.oib - Series 1-1 0021_cell12_0.211Fb0.789Tc_accuracy14.0306index52': 59, 'ROI3_01.oib - Series 1-1 0021_cell15_0.2512Fb0.7488Tc_accuracy27.6851index54': 52, 'ROI3_02.oib - Series 1-1 0001_cell3_1Fb0Tc_accuracy23.216index10': 83, 'ROI3_02.oib - Series 1-1 0006_cell2_1Fb0Tc_accuracy9.6752index6': 86, 'ROI4_01-1 0021_cell5_0.7957Fb0.2043Tc_accuracy4.9072index65': 86, 'ROI4_01-1 0025_cell3_0.8765Fb0.1235Tc_accuracy28.1694index34': 56, 'ROI4_01-1 0026_cell0_0.8038Fb0.1962Tc_accuracy18.8708index38': 52, 'ROI4_01-1 0026_cell4_0.8159Fb0.1841Tc_accuracy34.1523index41': 74, 'ROI4_01-1 0028_cell12_0.7781Fb0.2219Tc_accuracy10.2744index69': 51, 'ROI4_01-1 0029_cell18_0.6915Fb0.3085Tc_accuracy50.4092index61': 55, 'ROI4_01-1 0030_cell15_0.9255Fb0.07446Tc_accuracy86.0397index33': 54, 'ROI4_01-1 0030_cell2_0.9421Fb0.05788Tc_accuracy4.8176index35': 67, 'ROI4_01.oib - Series 1 0011_cell0_0.5592Fb0.4408Tc_accuracy7.9712index55': 82, 'ROI4_01.oib - Series 1 0011_cell6_0.5943Fb0.4057Tc_accuracy31.1473index36': 72, 'ROI4_01.oib - Series 1 0013_cell3_0.5327Fb0.4673Tc_accuracy39.6718index45': 52, 'ROI4_01.oib - Series 1 0018_cell12_0.5928Fb0.4072Tc_accuracy8.8109index90': 76, 'ROI4_01.oib - Series 1 0018_cell22_0.8316Fb0.1684Tc_accuracy52.2596index68': 58, 'ROI4_01.oib - Series 1 0018_cell3_0.9238Fb0.07625Tc_accuracy13.7437index66': 82, 'ROI4_01.oib - Series 1 0018_cell6_0.7413Fb0.2587Tc_accuracy18.7945index44': 55, 'ROI4_02.oib - Series 1-1 0012_cell15_0.3596Fb0.6404Tc_accuracy6.7286index33': 58, 'ROI4_02.oib - Series 1-1 0012_cell4_0.7164Fb0.2836Tc_accuracy52.1739index28': 53, 'ROI5_01-1 0001_cell23_1Fb0Tc_accuracy78.6694index37': 51, 'ROI5_01-1 0006_cell0_1Fb0Tc_accuracy39.127index8': 68, 'ROI5_01-1 0006_cell1_0.002653Fb0.9973Tc_accuracy4.8417index15': 54, 'ROI5_01-1 0007_cell11_0.9987Fb0.00134Tc_accuracy50.2586index35': 62, 'ROI5_01-1 0008_cell37_0.999Fb0.0009681Tc_accuracy49.2673index70': 63, 'ROI5_01-1 0008_cell8_0.8484Fb0.1516Tc_accuracy14.7885index61': 100, 'ROI5_01-1 0009_cell3_0.7683Fb0.2317Tc_accuracy29.0177index38': 82, 'ROI5_01-1 0009_cell4_0.8268Fb0.1732Tc_accuracy23.85index32': 69, 'ROI5_01-1 0012_cell2_0.6224Fb0.3776Tc_accuracy0.7478index34': 70, 'ROI5_01-1 0013_cell1_0.88Fb0.12Tc_accuracy70.7973index20': 54, 'ROI5_01-1 0014_cell8_0.7638Fb0.2362Tc_accuracy33.022index26': 62, 'ROI5_01-1 0015_cell24_0.8129Fb0.1871Tc_accuracy49.3753index59': 62, 'ROI5_01-1 0017_cell2_0.6161Fb0.3839Tc_accuracy13.1973index16': 54, 'ROI5_01-1 0018_cell10_0.7697Fb0.2303Tc_accuracy28.3165index51': 65, 'ROI5_01-1 0018_cell4_0.838Fb0.162Tc_accuracy16.5584index45': 101, 'ROI5_01-1 0019_cell3_0.9881Fb0.0119Tc_accuracy2.2329index45': 70, 'ROI5_01-1 0019_cell8_0.9271Fb0.07291Tc_accuracy10.5862index45': 102, 'ROI5_01-1 0020_cell3_0.6939Fb0.3061Tc_accuracy23.0489index26': 54, 'ROI5_01-1 0020_cell9_0.119Fb0.881Tc_accuracy18.654index56': 75, 'ROI5_01-1 0021_cell7_0.9981Fb0.001889Tc_accuracy78.3784index27': 80, 'ROI5_01-1 0022_cell40_0.8485Fb0.1515Tc_accuracy9.9923index59': 93, 'ROI5_01-1 0023_cell11_0.7905Fb0.2095Tc_accuracy29.7472index44': 107, 'ROI5_01-1 0024_cell11_0.3632Fb0.6368Tc_accuracy35.1819index34': 55, 'ROI5_01-1 0025_cell3_1Fb0Tc_accuracy18.5463index10': 109, 'ROI5_01-1 0026_cell1_0.9995Fb0.0005144Tc_accuracy22.3665index11': 71, 'ROI5_01-1 0027_cell7_0.7578Fb0.2422Tc_accuracy42.3139index17': 74, 'ROI5_01-1 0032_cell3_1Fb0Tc_accuracy18.5463index12': 109, 'ROI5_01.oib - Series 1-1 0002_cell18_0.6672Fb0.3328Tc_accuracy34.1882index51': 59, 'ROI5_01.oib - Series 1-1 0005_cell15_0.3781Fb0.6219Tc_accuracy42.3555index75': 57, 'ROI5_01.oib - Series 1-1 0006_cell14_0.5627Fb0.4373Tc_accuracy22.8126index114': 64, 'ROI5_01.oib - Series 1-1 0006_cell2_0.5195Fb0.4805Tc_accuracy34.5487index63': 61, 'ROI5_01.oib - Series 1-1 0021_cell3_0.1479Fb0.8521Tc_accuracy2.9688index12': 74, 'ROI5_02.oib - Series 1-1 0007_cell11_0.6015Fb0.3985Tc_accuracy77.5075index30': 55, 'ROI5_02.oib - Series 1-1 0008_cell11_0.7373Fb0.2627Tc_accuracy26.895index28': 74, 'ROI5_02.oib - Series 1-1 0008_cell13_0.3418Fb0.6582Tc_accuracy24.7909index48': 71, 'ROI5_02.oib - Series 1-1 0008_cell15_0.7063Fb0.2937Tc_accuracy3.3339index37': 64, 'ROI5_02.oib - Series 1-1 0008_cell2_0.2488Fb0.7512Tc_accuracy11.5528index39': 64, 'ROI5_02.oib - Series 1-1 0009_cell16_0.3289Fb0.6711Tc_accuracy24.5124index40': 64, 'ROI5_02.oib - Series 1-1 0012_cell12_0.5964Fb0.4036Tc_accuracy77.438index31': 55, 'ROI5_02.oib - Series 1-1 0012_cell5_0.6043Fb0.3957Tc_accuracy11.6441index31': 94, 'ROI5_02.oib - Series 1-1 0013_cell11_0.6511Fb0.3489Tc_accuracy26.9669index37': 73, 'ROI5_02.oib - Series 1-1 0013_cell15_0.3435Fb0.6565Tc_accuracy24.7135index58': 71, 'ROI5_02.oib - Series 1-1 0013_cell16_0.6143Fb0.3857Tc_accuracy11.5108index59': 78, 'ROI5_02.oib - Series 1-1 0013_cell24_0.4319Fb0.5681Tc_accuracy16.1843index65': 71, 'ROI5_02.oib - Series 1-1 0014_cell14_0.227Fb0.773Tc_accuracy24.6452index39': 64, 'ROI6_01-1 0027_cell3_0.7321Fb0.2679Tc_accuracy52.2363index21': 58, 'ROI6_01-1 0032_cell2_1Fb0Tc_accuracy17.0968index11': 59, 'ROI6_01.oib - Series 1-1 0010_cell13_0.4976Fb0.5024Tc_accuracy26.1278index54': 58, 'ROI6_01.oib - Series 1-1 0010_cell31_0.6581Fb0.3419Tc_accuracy33.4131index75': 55, 'ROI6_01.oib - Series 1-1 0010_cell41_0.5627Fb0.4373Tc_accuracy35.2172index75': 58, 'ROI6_01.oib - Series 1-1 0010_cell44_0.4333Fb0.5667Tc_accuracy8.8218index100': 61, 'ROI6_01.oib - Series 1-1 0011_cell31_0.5904Fb0.4096Tc_accuracy15.703index60': 52, 'ROI6_01.oib - Series 1-1 0011_cell41_0.4659Fb0.5341Tc_accuracy29.5366index67': 61, 'ROI6_01.oib - Series 1-1 0015_cell13_0.5646Fb0.4354Tc_accuracy36.0531index23': 56, 'ROI6_01.oib - Series 1-1 0015_cell4_0.6372Fb0.3628Tc_accuracy33.7889index23': 55, 'ROI6_01.oib - Series 1-1 0015_cell6_0Fb1Tc_accuracy5.9824index34': 78, 'ROI6_01.oib - Series 1-1 0016_cell8_0.0004552Fb0.9995Tc_accuracy31.5663index19': 60, 'ROI6_02.oib - Series 1-1 0012_cell0_0.5126Fb0.4874Tc_accuracy15.2471index22': 63, 'ROI6_02.oib - Series 1-1 0017_cell1_0.3353Fb0.6647Tc_accuracy7.9548index32': 69, 'ROI7_01-1 0005_cell1_1Fb0Tc_accuracy23.8621index20': 52, 'ROI7_01-1 0011_cell4_1Fb0Tc_accuracy34.0888index10': 72, 'ROI7_01-1 0018_cell7_1Fb0Tc_accuracy37.1827index17': 66, 'ROI7_01-1 0019_cell2_0.6096Fb0.3904Tc_accuracy26.9661index26': 54, 'ROI7_01-1 0020_cell9_0.4173Fb0.5827Tc_accuracy27.5387index51': 54, 'ROI7_01-1 0026_cell0_0.3304Fb0.6696Tc_accuracy5.4802index41': 52, 'ROI7_01-1 0027_cell3_0.5082Fb0.4918Tc_accuracy4.1008index29': 64, 'ROI7_01-1 0033_cell2_0.526Fb0.474Tc_accuracy24.6469index43': 60, 'ROI7_01-1 0033_cell4_0.3943Fb0.6057Tc_accuracy38.5173index22': 55, 'ROI7_01-1 0034_cell11_0.9056Fb0.09436Tc_accuracy50.1983index36': 66, 'ROI7_01-1 0034_cell2_0.3858Fb0.6142Tc_accuracy38.419index21': 54, 'ROI7_01-1 0040_cell6_0.8772Fb0.1228Tc_accuracy14.7442index25': 67, 'ROI7_01-1 0047_cell12_0.6881Fb0.3119Tc_accuracy15.2486index30': 66, 'ROI7_01.oib - Series 1-1 0013_cell6_0Fb1Tc_accuracy0.9394index10': 62, 'ROI7_02.oib - Series 1-1 0007_cell7_1Fb0Tc_accuracy3.7471index19': 58, 'ROI7_02.oib - Series 1-1 0008_cell6_0.9531Fb0.0469Tc_accuracy20.1303index24': 58, 'ROI7_02.oib - Series 1-1 0012_cell4_1Fb0Tc_accuracy3.7568index11': 57, 'ROI7_02.oib - Series 1-1 0013_cell8_0.935Fb0.06495Tc_accuracy17.2546index26': 58, 'ROI8_01-1 0001_cell0_1Fb0Tc_accuracy9.0409index3': 55, 'ROI8_01-1 0002_cell1_1Fb0Tc_accuracy9.0409index5': 55, 'ROI8_01-1 0011_cell9_0.8708Fb0.1292Tc_accuracy16.9197index22': 99, 'ROI8_01-1 0018_cell9_0.8634Fb0.1366Tc_accuracy14.3127index33': 94, 'ROI8_01-1 0026_cell9_0.4675Fb0.5325Tc_accuracy16.1738index31': 75, 'ROI8_01-1 0036_cell4_0Fb1Tc_accuracy14.5474index12': 54, 'ROI8_01-1 0045_cell6_0.9187Fb0.08129Tc_accuracy32.8431index11': 56, 'ROI8_01-1 0047_cell2_0.1027Fb0.8973Tc_accuracy9.4924index7': 65, 'ROI8_01.oib - Series 1-1 0003_cell1_1Fb0Tc_accuracy56.1258index3': 55, 'ROI8_01.oib - Series 1-1 0007_cell9_0.7618Fb0.2382Tc_accuracy15.5833index40': 81, 'ROI8_01.oib - Series 1-1 0008_cell13_1Fb0Tc_accuracy62.3244index19': 53, 'ROI9_01.oib - Series 1-1 0006_cell20_0.6469Fb0.3531Tc_accuracy29.3787index42': 52, 'ROI9_01.oib - Series 1-1 0010_cell26_0.7451Fb0.2549Tc_accuracy18.7777index51': 70, 'ROI9_01.oib - Series 1-1 0011_cell27_0.7284Fb0.2716Tc_accuracy33.2287index67': 61, 'ROI9_01.oib - Series 1-1 0012_cell6_0.7381Fb0.2619Tc_accuracy30.4652index62': 58, 'ROI9_01.oib - Series 1-1 0012_cell7_0.6546Fb0.3454Tc_accuracy45.4892index68': 54, 'ROI9_01.oib - Series 1-1 0015_cell9_0.8406Fb0.1594Tc_accuracy19.1375index29': 77, 'ROI9_01.oib - Series 1-1 0016_cell15_0.7212Fb0.2788Tc_accuracy23.4968index45': 60, 'ROI9_01.oib - Series 1-1 0017_cell15_0.792Fb0.208Tc_accuracy16.8204index68': 61, 'ROI9_01.oib - Series 1-1 0017_cell24_0.9024Fb0.09764Tc_accuracy37.6027index73': 55, 'ROI9_01.oib - Series 1-1 0017_cell5_0.1646Fb0.8354Tc_accuracy5.7429index86': 58, 'ROI9_01.oib - Series 1-1 0022_cell17_0.6308Fb0.3692Tc_accuracy6.7327index34': 72, 'ROI9_01.oib - Series 1-1 0022_cell20_0.2859Fb0.7141Tc_accuracy30.7509index53': 53, 'ROI9_01.oib - Series 1-1 0022_cell8_0.7339Fb0.2661Tc_accuracy15.9969index33': 59, 'ROI9_02.oib - Series 1-1 0006_cell2_0.7141Fb0.2859Tc_accuracy26.9001index34': 60, 'ROI9_02.oib - Series 1-1 0011_cell14_1Fb0Tc_accuracy43.0662index24': 55}
The strange values for y are {'ROI10_01.oib - Series 1-1 0001_cell5_0.7658Fb0.2342Tc_accuracy19.6944index37': 58, 'ROI10_01.oib - Series 1-1 0002_cell4_0.9508Fb0.04917Tc_accuracy16.987index24': 58, 'ROI1_01-1 0017_cell0_1Fb0Tc_accuracy18.5045index10': 54, 'ROI1_01-1 0018_cell8_0.7638Fb0.2362Tc_accuracy14.2019index38': 70, 'ROI1_01-1 0025_cell11_0.7785Fb0.2215Tc_accuracy61.9061index29': 62, 'ROI1_01-1 0025_cell19_0.991Fb0.008977Tc_accuracy54.227index31': 66, 'ROI1_01.oib - Series 1-1 0013_cell3_0.1611Fb0.8389Tc_accuracy23.2223index34': 64, 'ROI1_01.oib - Series 1-1 0013_cell8_0.9088Fb0.09118Tc_accuracy48.8504index28': 83, 'ROI1_01.oib - Series 1-1 0014_cell0_0.7105Fb0.2895Tc_accuracy17.0513index18': 90, 'ROI1_01.oib - Series 1-1 0014_cell2_0.9185Fb0.08148Tc_accuracy43.0786index33': 72, 'ROI1_01.oib - Series 1-1 0014_cell3_0.227Fb0.773Tc_accuracy23.9014index40': 64, 'ROI1_01.oib - Series 1-1 0015_cell3_0.9369Fb0.06313Tc_accuracy50.4062index23': 72, 'ROI1_01.oib - Series 1-1 0017_cell5_1Fb0Tc_accuracy15.1942index19': 77, 'ROI1_01.oib - Series 1-1 0018_cell9_0.77Fb0.23Tc_accuracy18.5052index24': 67, 'ROI1_01.oib - Series 1-1 0019_cell12_0.9126Fb0.08738Tc_accuracy1.2362index17': 98, 'ROI1_01.oib - Series 1-1 0020_cell2_0.9481Fb0.05195Tc_accuracy49.7759index31': 72, 'ROI1_01.oib - Series 1-1 0021_cell0_0.6892Fb0.3108Tc_accuracy14.5651index23': 72, 'ROI1_01.oib - Series 1-1 0021_cell4_0.7384Fb0.2616Tc_accuracy10.3712index24': 72, 'ROI1_01.oib - Series 1-1 0022_cell2_1Fb0Tc_accuracy13.8428index13': 91, 'ROI1_02.oib - Series 0007_cell4_0.6625Fb0.3375Tc_accuracy21.6905index42': 57, 'ROI1_02.oib - Series 0007_cell7_0.6853Fb0.3147Tc_accuracy36.4385index45': 61, 'ROI1_02.oib - Series 0012_cell1_0.7733Fb0.2267Tc_accuracy10.0514index45': 64, 'ROI1_02.oib - Series 0012_cell6_0.6625Fb0.3375Tc_accuracy21.8217index40': 58, 'ROI1_02.oib - Series 0013_cell16_0.2189Fb0.7811Tc_accuracy4.2002index61': 68, 'ROI2_01-1 0007_cell2_0.9113Fb0.08872Tc_accuracy30.6453index48': 61, 'ROI2_01-1 0020_cell2_0.6511Fb0.3489Tc_accuracy4.03index38': 87, 'ROI2_01-1 0031_cell9_0.726Fb0.274Tc_accuracy22.7933index35': 61, 'ROI2_01-1 0034_cell1_1Fb0Tc_accuracy36.9678index18': 52, 'ROI2_01-1 0034_cell6_1Fb0Tc_accuracy43.5447index22': 56, 'ROI2_01.oib - Series 1-1 0008_cell16_0.1226Fb0.8774Tc_accuracy5.9905index44': 52, 'ROI2_01.oib - Series 1-1 0008_cell7_0.9511Fb0.04894Tc_accuracy24.0327index24': 52, 'ROI2_01.oib - Series 1-1 0012_cell4_0.4308Fb0.5692Tc_accuracy39.7292index30': 58, 'ROI2_01.oib - Series 1-1 0013_cell5_0.2956Fb0.7044Tc_accuracy29.5626index57': 53, 'ROI2_01.oib - Series 1-1 0018_cell7_0.5757Fb0.4243Tc_accuracy38.2652index38': 58, 'ROI2_02.oib - Series 1-1 (cropped) 0001_cell7_0.5885Fb0.4115Tc_accuracy31.6285index35': 60, 'ROI2_02.oib - Series 1-1 (cropped) 0002_cell9_0Fb1Tc_accuracy30.7565index28': 60, 'ROI2_02.oib - Series 1-1 (cropped) 0005_cell5_0.6162Fb0.3838Tc_accuracy6.3765index15': 125, 'ROI2_02.oib - Series 1-1 (cropped) 0006_cell3_0.6163Fb0.3837Tc_accuracy4.3986index24': 90, 'ROI2_02.oib - Series 1-1 (cropped) 0011_cell3_0Fb1Tc_accuracy0.989index19': 72, 'ROI2_02.oib - Series 1-1 (cropped) 0013_cell3_0.8171Fb0.1829Tc_accuracy26.214index15': 68, 'ROI2_02.oib - Series 1-1 (cropped) 0014_cell1_1Fb0Tc_accuracy8.8602index6': 76, 'ROI2_02.oib - Series 1-1 (cropped) 0016_cell4_0.575Fb0.425Tc_accuracy5.2412index27': 65, 'ROI2_02.oib - Series 1-1 (cropped) 0016_cell8_0Fb1Tc_accuracy30.4572index28': 66, 'ROI2_02.oib - Series 1-1 (cropped) 0017_cell5_0.1852Fb0.8148Tc_accuracy41.3422index22': 66, 'ROI2_02.oib - Series 1-1 (cropped) 0017_cell7_1Fb0Tc_accuracy26.2231index15': 58, 'ROI3_01-1 0031_cell6_0.829Fb0.171Tc_accuracy13.4426index44': 112, 'ROI3_01-1 0043_cell0_1Fb0Tc_accuracy31.0317index8': 63, 'ROI3_01.oib - Series 1-1 0008_cell1_0.7266Fb0.2734Tc_accuracy41.4689index17': 51, 'ROI3_01.oib - Series 1-1 0015_cell14_0.9994Fb0.0005548Tc_accuracy30.7207index41': 66, 'ROI3_01.oib - Series 1-1 0015_cell21_0.5558Fb0.4442Tc_accuracy16.6556index46': 59, 'ROI3_01.oib - Series 1-1 0016_cell11_0.6451Fb0.3549Tc_accuracy13.283index27': 62, 'ROI3_01.oib - Series 1-1 0027_cell2_0.395Fb0.605Tc_accuracy54.8955index16': 51, 'ROI3_01.oib - Series 1-1 0028_cell1_0.3893Fb0.6107Tc_accuracy58.3804index14': 51, 'ROI3_02.oib - Series 1-1 0002_cell7_1Fb0Tc_accuracy35.5201index14': 56, 'ROI3_02.oib - Series 1-1 0006_cell2_1Fb0Tc_accuracy9.6752index6': 132, 'ROI3_02.oib - Series 1-1 0007_cell4_0.1948Fb0.8052Tc_accuracy41.7212index30': 57, 'ROI3_02.oib - Series 1-1 0007_cell9_0.6116Fb0.3884Tc_accuracy43.7042index24': 51, 'ROI3_02.oib - Series 1-1 0012_cell6_0.1298Fb0.8702Tc_accuracy42.8745index37': 57, 'ROI4_01-1 0008_cell2_1Fb0Tc_accuracy3.9366index12': 56, 'ROI4_01-1 0015_cell4_1Fb0Tc_accuracy26.4059index23': 56, 'ROI4_01-1 0021_cell5_0.7957Fb0.2043Tc_accuracy4.9072index65': 70, 'ROI4_01-1 0025_cell3_0.8765Fb0.1235Tc_accuracy28.1694index34': 85, 'ROI4_01-1 0026_cell0_0.8038Fb0.1962Tc_accuracy18.8708index38': 51, 'ROI4_01-1 0026_cell4_0.8159Fb0.1841Tc_accuracy34.1523index41': 59, 'ROI4_01-1 0028_cell12_0.7781Fb0.2219Tc_accuracy10.2744index69': 63, 'ROI4_01-1 0030_cell2_0.9421Fb0.05788Tc_accuracy4.8176index35': 56, 'ROI4_01-1 0043_cell5_0.1834Fb0.8166Tc_accuracy15.7573index29': 58, 'ROI4_01.oib - Series 1 0003_cell16_0.2706Fb0.7294Tc_accuracy29.2996index45': 52, 'ROI4_01.oib - Series 1 0011_cell0_0.5592Fb0.4408Tc_accuracy7.9712index55': 54, 'ROI4_01.oib - Series 1 0012_cell27_0.622Fb0.378Tc_accuracy26.962index68': 57, 'ROI4_01.oib - Series 1 0012_cell28_0.2585Fb0.7415Tc_accuracy32.672index100': 51, 'ROI4_01.oib - Series 1 0013_cell10_0.7438Fb0.2562Tc_accuracy19.8991index56': 58, 'ROI4_01.oib - Series 1 0013_cell11_0.6192Fb0.3808Tc_accuracy66.675index56': 51, 'ROI4_01.oib - Series 1 0017_cell19_0.6114Fb0.3886Tc_accuracy27.4156index41': 53, 'ROI4_01.oib - Series 1 0017_cell20_0.2447Fb0.7553Tc_accuracy35.9257index66': 51, 'ROI4_01.oib - Series 1 0018_cell12_0.5928Fb0.4072Tc_accuracy8.8109index90': 77, 'ROI4_01.oib - Series 1 0018_cell22_0.8316Fb0.1684Tc_accuracy52.2596index68': 62, 'ROI4_01.oib - Series 1 0018_cell3_0.9238Fb0.07625Tc_accuracy13.7437index66': 78, 'ROI4_01.oib - Series 1 0018_cell8_0.5903Fb0.4097Tc_accuracy62.6715index55': 51, 'ROI4_02.oib - Series 1-1 0010_cell6_0.0007974Fb0.9992Tc_accuracy40.801index21': 60, 'ROI4_02.oib - Series 1-1 0012_cell4_0.7164Fb0.2836Tc_accuracy52.1739index28': 65, 'ROI4_02.oib - Series 1-1 0016_cell3_0.9335Fb0.06651Tc_accuracy17.2245index19': 70, 'ROI5_01-1 0001_cell12_0.5301Fb0.4699Tc_accuracy23.4109index41': 51, 'ROI5_01-1 0003_cell2_1Fb0Tc_accuracy29.4941index15': 66, 'ROI5_01-1 0007_cell11_0.9987Fb0.00134Tc_accuracy50.2586index35': 61, 'ROI5_01-1 0008_cell37_0.999Fb0.0009681Tc_accuracy49.2673index70': 61, 'ROI5_01-1 0008_cell8_0.8484Fb0.1516Tc_accuracy14.7885index61': 88, 'ROI5_01-1 0009_cell4_0.8268Fb0.1732Tc_accuracy23.85index32': 76, 'ROI5_01-1 0014_cell0_1Fb0Tc_accuracy51.8246index22': 52, 'ROI5_01-1 0014_cell8_0.7638Fb0.2362Tc_accuracy33.022index26': 61, 'ROI5_01-1 0015_cell24_0.8129Fb0.1871Tc_accuracy49.3753index59': 60, 'ROI5_01-1 0017_cell2_0.6161Fb0.3839Tc_accuracy13.1973index16': 68, 'ROI5_01-1 0018_cell10_0.7697Fb0.2303Tc_accuracy28.3165index51': 54, 'ROI5_01-1 0018_cell4_0.838Fb0.162Tc_accuracy16.5584index45': 77, 'ROI5_01-1 0019_cell37_0.4901Fb0.5099Tc_accuracy24.7668index92': 52, 'ROI5_01-1 0019_cell4_0.2471Fb0.7529Tc_accuracy15.8263index71': 55, 'ROI5_01-1 0019_cell8_0.9271Fb0.07291Tc_accuracy10.5862index45': 97, 'ROI5_01-1 0022_cell40_0.8485Fb0.1515Tc_accuracy9.9923index59': 87, 'ROI5_01-1 0023_cell11_0.7905Fb0.2095Tc_accuracy29.7472index44': 52, 'ROI5_01-1 0025_cell3_1Fb0Tc_accuracy18.5463index10': 54, 'ROI5_01-1 0026_cell1_0.9995Fb0.0005144Tc_accuracy22.3665index11': 66, 'ROI5_01-1 0026_cell27_0.5147Fb0.4853Tc_accuracy26.9493index79': 52, 'ROI5_01-1 0032_cell1_0.3746Fb0.6254Tc_accuracy33.6351index40': 51, 'ROI5_01-1 0032_cell3_1Fb0Tc_accuracy18.5463index12': 54, 'ROI5_01-1 0033_cell0_0.4032Fb0.5968Tc_accuracy26.2683index31': 54, 'ROI5_01-1 0046_cell1_1Fb0Tc_accuracy81.1255index5': 63, 'ROI5_01.oib - Series 1-1 0000_cell1_0.4088Fb0.5912Tc_accuracy23.5317index11': 56, 'ROI5_01.oib - Series 1-1 0001_cell22_0.4248Fb0.5752Tc_accuracy27.2067index51': 56, 'ROI5_01.oib - Series 1-1 0002_cell18_0.6672Fb0.3328Tc_accuracy34.1882index51': 58, 'ROI5_01.oib - Series 1-1 0005_cell7_0Fb1Tc_accuracy22.7362index59': 55, 'ROI5_01.oib - Series 1-1 0006_cell2_0.5195Fb0.4805Tc_accuracy34.5487index63': 109, 'ROI5_01.oib - Series 1-1 0006_cell44_0.2642Fb0.7358Tc_accuracy20.5988index138': 56, 'ROI5_01.oib - Series 1-1 0007_cell10_0.7722Fb0.2278Tc_accuracy41.784index46': 65, 'ROI5_01.oib - Series 1-1 0011_cell42_0.4916Fb0.5084Tc_accuracy24.8141index76': 56, 'ROI5_01.oib - Series 1-1 0021_cell3_0.1479Fb0.8521Tc_accuracy2.9688index12': 94, 'ROI5_02.oib - Series 1-1 0007_cell6_0.8887Fb0.1113Tc_accuracy42.8275index26': 51, 'ROI5_02.oib - Series 1-1 0008_cell13_0.3418Fb0.6582Tc_accuracy24.7909index48': 56, 'ROI5_02.oib - Series 1-1 0008_cell2_0.2488Fb0.7512Tc_accuracy11.5528index39': 58, 'ROI5_02.oib - Series 1-1 0009_cell16_0.3289Fb0.6711Tc_accuracy24.5124index40': 56, 'ROI5_02.oib - Series 1-1 0012_cell5_0.6043Fb0.3957Tc_accuracy11.6441index31': 58, 'ROI5_02.oib - Series 1-1 0013_cell15_0.3435Fb0.6565Tc_accuracy24.7135index58': 56, 'ROI5_02.oib - Series 1-1 0013_cell16_0.6143Fb0.3857Tc_accuracy11.5108index59': 63, 'ROI5_02.oib - Series 1-1 0013_cell24_0.4319Fb0.5681Tc_accuracy16.1843index65': 101, 'ROI5_02.oib - Series 1-1 0014_cell14_0.227Fb0.773Tc_accuracy24.6452index39': 56, 'ROI5_02.oib - Series 1-1 0018_cell13_0Fb1Tc_accuracy26.8325index33': 56, 'ROI6_01-1 0022_cell20_0.1659Fb0.8341Tc_accuracy4.7805index52': 53, 'ROI6_01.oib - Series 1-1 0010_cell13_0.4976Fb0.5024Tc_accuracy26.1278index54': 80, 'ROI6_01.oib - Series 1-1 0015_cell6_0Fb1Tc_accuracy5.9824index34': 52, 'ROI6_02.oib - Series 1-1 0002_cell1_0.4742Fb0.5258Tc_accuracy18.6628index19': 52, 'ROI6_02.oib - Series 1-1 0006_cell14_0Fb1Tc_accuracy16.3602index47': 63, 'ROI6_02.oib - Series 1-1 0006_cell1_0.6941Fb0.3059Tc_accuracy42.0789index24': 51, 'ROI6_02.oib - Series 1-1 0007_cell1_0.8538Fb0.1462Tc_accuracy38.3605index16': 52, 'ROI6_02.oib - Series 1-1 0011_cell0_0.628Fb0.372Tc_accuracy30.0732index21': 59, 'ROI6_02.oib - Series 1-1 0012_cell0_0.5126Fb0.4874Tc_accuracy15.2471index22': 60, 'ROI6_02.oib - Series 1-1 0017_cell1_0.3353Fb0.6647Tc_accuracy7.9548index32': 61, 'ROI7_01-1 0004_cell22_1Fb0Tc_accuracy30.8824index51': 56, 'ROI7_01-1 0005_cell14_1Fb0Tc_accuracy31.0168index26': 55, 'ROI7_01-1 0010_cell19_0Fb1Tc_accuracy21.8266index67': 51, 'ROI7_01-1 0020_cell16_0.6953Fb0.3047Tc_accuracy40.2687index26': 60, 'ROI7_01-1 0026_cell0_0.3304Fb0.6696Tc_accuracy5.4802index41': 76, 'ROI7_01-1 0027_cell13_0.6996Fb0.3004Tc_accuracy39.955index29': 61, 'ROI7_01-1 0027_cell3_0.5082Fb0.4918Tc_accuracy4.1008index29': 81, 'ROI7_01-1 0032_cell5_0.9497Fb0.05031Tc_accuracy51.1407index20': 51, 'ROI7_01-1 0033_cell2_0.9781Fb0.02192Tc_accuracy29.6961index21': 54, 'ROI7_01-1 0033_cell6_0.1111Fb0.8889Tc_accuracy2.6378index38': 51, 'ROI7_01-1 0034_cell2_0.9821Fb0.01791Tc_accuracy4.3527index27': 54, 'ROI7_01-1 0040_cell6_0.8772Fb0.1228Tc_accuracy14.7442index25': 65, 'ROI7_01.oib - Series 1-1 0013_cell6_0Fb1Tc_accuracy0.9394index10': 66, 'ROI7_02.oib - Series 1-1 0007_cell7_1Fb0Tc_accuracy3.7471index19': 54, 'ROI7_02.oib - Series 1-1 0008_cell2_1Fb0Tc_accuracy34.0928index21': 62, 'ROI7_02.oib - Series 1-1 0008_cell6_0.9531Fb0.0469Tc_accuracy20.1303index24': 54, 'ROI7_02.oib - Series 1-1 0009_cell3_1Fb0Tc_accuracy14.7233index12': 62, 'ROI7_02.oib - Series 1-1 0012_cell4_1Fb0Tc_accuracy3.7568index11': 53, 'ROI7_02.oib - Series 1-1 0013_cell8_0.935Fb0.06495Tc_accuracy17.2546index26': 53, 'ROI8_01-1 0001_cell0_1Fb0Tc_accuracy9.0409index3': 52, 'ROI8_01-1 0002_cell1_1Fb0Tc_accuracy9.0409index5': 52, 'ROI8_01-1 0011_cell9_0.8708Fb0.1292Tc_accuracy16.9197index22': 57, 'ROI8_01-1 0019_cell5_0.5496Fb0.4504Tc_accuracy30.4275index26': 62, 'ROI8_01-1 0026_cell9_0.4675Fb0.5325Tc_accuracy16.1738index31': 79, 'ROI8_01-1 0036_cell4_0Fb1Tc_accuracy14.5474index12': 65, 'ROI8_01-1 0040_cell4_1Fb0Tc_accuracy14.469index8': 57, 'ROI8_01-1 0045_cell6_0.9187Fb0.08129Tc_accuracy32.8431index11': 51, 'ROI8_01-1 0047_cell2_0.1027Fb0.8973Tc_accuracy9.4924index7': 74, 'ROI8_01.oib - Series 1-1 0007_cell9_0.7618Fb0.2382Tc_accuracy15.5833index40': 98, 'ROI9_01.oib - Series 1-1 0006_cell20_0.6469Fb0.3531Tc_accuracy29.3787index42': 64, 'ROI9_01.oib - Series 1-1 0012_cell6_0.7381Fb0.2619Tc_accuracy30.4652index62': 69, 'ROI9_01.oib - Series 1-1 0015_cell9_0.8406Fb0.1594Tc_accuracy19.1375index29': 63, 'ROI9_01.oib - Series 1-1 0016_cell15_0.7212Fb0.2788Tc_accuracy23.4968index45': 85, 'ROI9_01.oib - Series 1-1 0017_cell24_0.9024Fb0.09764Tc_accuracy37.6027index73': 65, 'ROI9_01.oib - Series 1-1 0017_cell5_0.1646Fb0.8354Tc_accuracy5.7429index86': 69, 'ROI9_02.oib - Series 1-1 0006_cell2_0.7141Fb0.2859Tc_accuracy26.9001index34': 62}
The strange values for z are {'ROI10_01.oib - Series 1-1 0001_cell2_0.81Fb0.19Tc_accuracy7.2851index43': 22, 'ROI10_01.oib - Series 1-1 0006_cell18_0.2851Fb0.7149Tc_accuracy20.6897index103': 22, 'ROI1_01-1 0000_cell0_1Fb0Tc_accuracy34.0263index2': 21, 'ROI1_01-1 0001_cell0_1Fb0Tc_accuracy34.0263index5': 21, 'ROI1_01-1 0018_cell8_0.7638Fb0.2362Tc_accuracy14.2019index38': 25, 'ROI1_01-1 0033_cell2_0.2726Fb0.7274Tc_accuracy12.5406index40': 21, 'ROI1_01.oib - Series 1-1 0013_cell14_0.2272Fb0.7728Tc_accuracy25.3304index33': 23, 'ROI1_01.oib - Series 1-1 0018_cell9_0.77Fb0.23Tc_accuracy18.5052index24': 23, 'ROI2_01-1 0007_cell2_0.9113Fb0.08872Tc_accuracy30.6453index48': 21, 'ROI3_01-1 0024_cell0_0.8724Fb0.1276Tc_accuracy19.1201index24': 33, 'ROI3_01-1 0030_cell0_0.8594Fb0.1406Tc_accuracy20.2521index27': 22, 'ROI4_01-1 0021_cell5_0.7957Fb0.2043Tc_accuracy4.9072index65': 23, 'ROI4_01-1 0025_cell3_0.8765Fb0.1235Tc_accuracy28.1694index34': 23, 'ROI4_01-1 0026_cell0_0.8038Fb0.1962Tc_accuracy18.8708index38': 24, 'ROI4_01-1 0030_cell2_0.9421Fb0.05788Tc_accuracy4.8176index35': 21, 'ROI4_01.oib - Series 1 0018_cell12_0.5928Fb0.4072Tc_accuracy8.8109index90': 23, 'ROI4_01.oib - Series 1 0018_cell3_0.9238Fb0.07625Tc_accuracy13.7437index66': 27, 'ROI4_02.oib - Series 1-1 0016_cell3_0.9335Fb0.06651Tc_accuracy17.2245index19': 21, 'ROI5_01-1 0008_cell8_0.8484Fb0.1516Tc_accuracy14.7885index61': 21, 'ROI5_01-1 0009_cell3_0.7683Fb0.2317Tc_accuracy29.0177index38': 22, 'ROI5_01-1 0009_cell4_0.8268Fb0.1732Tc_accuracy23.85index32': 23, 'ROI5_01-1 0023_cell1_0.758Fb0.242Tc_accuracy34.1685index44': 26, 'ROI5_01-1 0030_cell1_0.7084Fb0.2916Tc_accuracy19.9879index58': 26, 'ROI5_01-1 0035_cell0_1Fb0Tc_accuracy56.1531index2': 21, 'ROI5_01.oib - Series 1-1 0005_cell9_0.2543Fb0.7457Tc_accuracy25.2354index61': 21, 'ROI5_01.oib - Series 1-1 0006_cell2_0.5195Fb0.4805Tc_accuracy34.5487index63': 21, 'ROI5_01.oib - Series 1-1 0006_cell48_0.2431Fb0.7569Tc_accuracy23.2706index141': 21, 'ROI5_01.oib - Series 1-1 0010_cell17_0.4434Fb0.5566Tc_accuracy23.3078index66': 21, 'ROI6_01-1 0031_cell8_1Fb0Tc_accuracy23.932index40': 21, 'ROI6_01.oib - Series 1-1 0010_cell13_0.4976Fb0.5024Tc_accuracy26.1278index54': 22, 'ROI6_01.oib - Series 1-1 0010_cell20_0.7012Fb0.2988Tc_accuracy21.5124index73': 21, 'ROI6_01.oib - Series 1-1 0011_cell17_0.6124Fb0.3876Tc_accuracy18.4889index51': 21, 'ROI7_01-1 0023_cell14_0.8069Fb0.1931Tc_accuracy28.4351index39': 21, 'ROI7_01-1 0030_cell11_1Fb0Tc_accuracy26.8508index37': 21, 'ROI8_01-1 0012_cell0_0.1991Fb0.8009Tc_accuracy30.1766index4': 23, 'ROI9_01.oib - Series 1-1 0011_cell27_0.7284Fb0.2716Tc_accuracy33.2287index67': 23, 'ROI9_01.oib - Series 1-1 0012_cell6_0.7381Fb0.2619Tc_accuracy30.4652index62': 22, 'ROI9_01.oib - Series 1-1 0016_cell15_0.7212Fb0.2788Tc_accuracy23.4968index45': 23, 'ROI9_01.oib - Series 1-1 0017_cell5_0.1646Fb0.8354Tc_accuracy5.7429index86': 22, 'ROI9_01.oib - Series 1-1 0022_cell17_0.6308Fb0.3692Tc_accuracy6.7327index34': 22}


Process finished with exit code 0

'''




import os
import random
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import shutil
import glob
from pathlib import Path
import re

'''
def create_label_matrix(the_directory_name):
    for path in Path(starting_directory).rglob('*.tif'):
        if "accuracy" in path.name:
            print(path.name)
            x = str(path.name)
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
                            fibroblast_counter = fibroblast_counter + 1
                        else:
                            cancer_cell_counter = cancer_cell_counter + 1

    print("Total number of fibroblasts is " + str(fibroblast_counter))
    print("Total number of cancer cells is " + str(cancer_cell_counter))
'''

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)
def stack_and_crop(array_to_crop,array_to_stack_on):
    keeper_rows = []
    keeper_columns = []
    for y in range(len(array_to_crop)):
        for x in range(len(array_to_crop[0])):
            if array_to_crop[y][x] != 0:
                if not (y in keeper_rows):
                    keeper_rows.append(y)
                if not (x in keeper_columns):
                    keeper_columns.append(x)
    minimum_row = min(keeper_rows)
    minimum_column = min(keeper_columns)
    maximum_row = max(keeper_rows)
    maximum_column = max(keeper_columns)
    cropped_arr1 = np.zeros((maximum_row-minimum_row+1, maximum_column-minimum_column+1))
    counterY = 0
    counterX = 0
    for y in range(minimum_row, maximum_row + 1):
        for x in range(minimum_column, maximum_column + 1):
            cropped_arr1[counterY][counterX] = array_to_crop[y][x]
            counterX = counterX + 1
        counterX = 0
        counterY = counterY + 1
    counterY = 0
    counterX = 0
    #print(cropped_arr1)
    array_to_stack_on.append(cropped_arr1)
    return array_to_stack_on

def find_strange_values(the_list, dir):
    global strange_y_values_dictionary, strange_x_values_dictionary, strange_z_values_dictionary
    max_z = len(the_list)
    max_y = 0
    max_x = 0
    for z in range(len(the_list)):
        if len(the_list[z]) > max_y:
            max_y = len(the_list[z])
        for y in range(len(the_list[z])):
            if len(the_list[z][y]) > max_x:
                max_x = len(the_list[z][y])
    if max_z > 20:
        strange_z_values_dictionary[dir] = max_z
    if max_y > 50:
        strange_y_values_dictionary[dir] = max_y
    if max_x > 50:
        strange_x_values_dictionary[dir] = max_x


def get_the_standard(the_list, dir):
    global true_maximum_z, true_maximum_x, true_maximum_y, directory_true_maximum_z, directory_true_maximum_y, directory_true_maximum_x
    max_z = len(the_list)
    max_y = 0
    max_x = 0
    for z in range(len(the_list)):
        if len(the_list[z]) > max_y:
            max_y = len(the_list[z])
        for y in range(len(the_list[z])):
            if len(the_list[z][y]) > max_x:
                max_x = len(the_list[z][y])
    if max_z > true_maximum_z:
        true_maximum_z = max_z
        directory_true_maximum_z = dir
    if max_y > true_maximum_y:
        true_maximum_y = max_y
        directory_true_maximum_y = dir
    if max_z > true_maximum_x:
        true_maximum_x = max_x
        directory_true_maximum_x = dir


# Pads the matrices in the list based on the maximum length and width of the matrices in the list.
def pad_matrices_in_list(the_list):
    max_y = 0
    max_x = 0
    for z in range(len(the_list)):
        if len(the_list[z]) > max_y:
            max_y = len(the_list[z])
        for y in range(len(the_list[z])):
            if len(the_list[z][y]) > max_x:
                max_x = len(the_list[z][y])
    padding_matrix = np.zeros((len(the_list),max_y,max_x))
    for z in range(len(the_list)):
        for y in range(len(the_list[z])):
            for x in range(len(the_list[z][y])):
                padding_matrix[z][y][x] = the_list[z][y][x]
    return padding_matrix

# Pads the matrices in the list based on a standard size.
# (SHOULD NOT BE USED IF THE STANDARD SIZE IS SMALLER THAN ANY OF THE MATRICES IN IT)
def pad_matrices_in_list_to_standard(the_list, tuple_size):
    padding_matrix = np.zeros((tuple_size))
    for z in range(len(the_list)):
        for y in range(len(the_list[z])):
            for x in range(len(the_list[z][y])):
                padding_matrix[y][x] = the_list[z][y][x]
    return padding_matrix


# The 5 3D arrays off this data
DAPI_3D_array = []
#Fibroblast_3D_array = []
#Cancer_3D_array = []
Reflection_3D_array = []
Transmission_brightfield_3D_array = []

strange_x_values_dictionary = {}
strange_y_values_dictionary = {}
strange_z_values_dictionary = {}
true_maximum_z = 0
directory_true_maximum_z = ""
true_maximum_y = 0
directory_true_maximum_y = ""
true_maximum_x = 0
directory_true_maximum_x = ""


for root, dirs, files in os.walk(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\Macro V5 images\20X (no standard) Images Macro V5\step2 - 2D matrices - Copy"):
    for dir in dirs:
        # Begin searching each directory for the 2D PNGs
        print("Now searching directory " + str(dir))
        for path in Path(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Image_Data\Macro V5 images\20X (no standard) Images Macro V5\step2 - 2D matrices - Copy",dir)).rglob("*.NPY"):
            x = str(path.name)
            if x[0] == "C":
                if x[1] == "1":
                    print("This path has a C1 image: " + path.name)
                    arr1 = np.load(str(path))
                    #print(arr1)
                    stack_and_crop(arr1,DAPI_3D_array)
                if x[1] == "2":
                    print("This path has a C2 image: " + path.name)
                    #arr2 = np.load(str(path))
                    #print(arr2)
                    #stack_and_crop(arr2, Fibroblast_3D_array)
                if x[1] == "3":
                    print("This path has a C3 image: " + path.name)
                    #arr3 = np.load(str(path))
                    #print(arr3)
                    #stack_and_crop(arr3, Cancer_3D_array)
                if x[1] == "4":
                    print("This path has a C4 image: " + path.name)
                    arr4 = np.load(str(path))
                    #print(arr4)
                    stack_and_crop(arr4, Reflection_3D_array)
                if x[1] == "5":
                    print("This path has a C5 image: " + path.name)
                    arr5 = np.load(str(path))
                    #print(arr5)
                    stack_and_crop(arr5, Transmission_brightfield_3D_array)

        # Padding the matrices
        find_strange_values(DAPI_3D_array, dir)
        #Fibroblast_3D_array = pad_matrices_in_list(Fibroblast_3D_array)
        #Cancer_3D_array = pad_matrices_in_list(Cancer_3D_array)
        find_strange_values(Reflection_3D_array, dir)
        find_strange_values(Transmission_brightfield_3D_array, dir)



        # reset the 3D matrices after saving them.
        DAPI_3D_array = []
        # Fibroblast_3D_array = []
        # Cancer_3D_array = []
        Reflection_3D_array = []
        Transmission_brightfield_3D_array = []

print("The strange values for x are " + str(strange_x_values_dictionary))
print("The strange values for y are " + str(strange_y_values_dictionary))
print("The strange values for z are " + str(strange_z_values_dictionary))
'''
DAPI_3D_array = np.array(DAPI_3D_array)
Fibroblast_3D_array = np.array(Fibroblast_3D_array)
Cancer_3D_array = np.array(Cancer_3D_array)
Reflection_3D_array = np.array(Reflection_3D_array)
Transmission_brightfield_3D_array = np.array(Transmission_brightfield_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\DAPI_3D_array",DAPI_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Fibroblast_3D_array",Fibroblast_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Cancer_3D_array",Cancer_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Reflection_3D_array",Reflection_3D_array)
np.save(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Cropping results\Attempt 2\Transmission_brightfield_3D_array",Transmission_brightfield_3D_array )
'''



'''keeper_rows = []
                    keeper_columns = []
                    for y in range(len(arr1)):
                        for x in range(len(arr1[0])):
                            if arr1[y][x] != 0:
                                if not (y in keeper_rows):
                                    keeper_rows.append(y)
                                if not (x in keeper_columns):
                                    keeper_columns.append(x)
                    cropped_arr1 = np.zeros(len(keeper_rows),len(keeper_columns))
                    minimum_row = min(keeper_rows)
                    minimum_column = min(keeper_columns)
                    maximum_row = max(keeper_rows)
                    maximum_column = max(keeper_columns)
                    for y in range(minimum_row,maximum_row+1):
                        for x in range(minimum_column,maximum_column+1):
                            cropped_arr1 = arr1[y][x]
                    print(cropped_arr1)
                    DAPI_3D_array.append(cropped_arr1)'''