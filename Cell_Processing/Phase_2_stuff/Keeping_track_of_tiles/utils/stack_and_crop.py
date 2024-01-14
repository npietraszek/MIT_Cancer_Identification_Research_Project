import numpy as np

'''
Dual purpose function that both stacks the 2D arrays and crops out the empty space around the cells.
Takes a array_to_crop and crops out the empty space around the cells, then stacks it on top of the array_to_stack_on.
Does this by checking if the rows or columns have any nonzero values in them. All rows contained between the maximum row and minimum row,
and all columns between the maximum column and minimum column are kept.
Doing this removes all empty space surrounding the cells.
Then, after this is done, takes the cropped array and appends it to the array_to_stack_on.

Parameters
----------
array_to_crop : 2D numpy array
    The array to crop.
array_to_stack_on : 3D numpy array
    The array to stack the cropped array on top of once the cropping is finished.

Returns
-------
array_to_stack_on : 3D numpy array
    The array with the cropped array stacked on top of it.
'''
def stack_and_crop(array_to_crop,array_to_stack_on):
    # Check what rows and columns have nonzero values in them.
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
    # Keep only the rows and columns between the maximum and minimum rows and columns with nonzero values in them. 
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
    # Append the cropped array to the array_to_stack_on.
    array_to_stack_on.append(cropped_arr1)
    return array_to_stack_on