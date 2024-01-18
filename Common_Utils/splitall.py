import os

'''
Function for splitting a path into its parts for easier processing.
Useful for tracing back certain directories.
Parameters
----------
path : string
    The path to split.
Returns
-------
allparts : list
    The path split into its parts.
'''
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts