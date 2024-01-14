import os

'''
Simple function to check if a directory exists, and if it doesn't, create it.
Parameters
----------
directory : string
    The directory to check for.
Returns
-------
None.
'''
def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created a missing folder at " + directory)