import re
'''
Support function to find a cell's accuracy number from its file name.
Parameters
----------
x : str
    The file name of the cell.

Returns
-------
accuracy_value : float
    The accuracy value of the cell.
'''
def find_accuracy_number(x:str):
    accuracy_index = x.find("accuracy")
    if accuracy_index != -1:
        letters_after_accuracy = x[accuracy_index+8:]
        list_of_numbers = ([float(s) for s in re.findall(r'-?\d+\.?\d*', letters_after_accuracy)])
        accuracy_value = list_of_numbers[0]
        return accuracy_value
    return -1