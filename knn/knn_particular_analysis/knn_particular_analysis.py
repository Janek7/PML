# Particular Data Analysis (with kNN) [1.2]

# Params
import math

inputfilename = 'iris.data'  # filename input data
outputfilename = inputfilename + '.erg'  # filename output data
inputseparator = ','  # separator for csv columns
labelcolumn = 4  # column of label
columnfilter = [0, 1, 2, 3]  # columns with features
columnflen = len(columnfilter)  # count of features in data
featurecols = [0, 1, 2, 3]  # selected Features
linefilter = '[]'  # linefilter: lines to ignore
firstrelpred = 10  # count of first neighbours in outputlist
rpdigits = 4  # relative prediction: number of digits after decimal point
dnnrange = [0, 1, 2, 3, 4]  # show different nearest neighbour range
sepaprod = 'lambda x,y: x*y'
printseparator = '*' * 80 + '\n'

# Header in Outputfile
print('Particular Data Analysis (with kNN) [1.2]\n')
print('Selected features:', featurecols)
print('Ignored lines:', linefilter)
print(printseparator + '\n')

# Load Data
lines = [line.split(',') for line in open('iris.data', 'r').read().split('\n')]
line_dict = {idx: line for idx, line in enumerate(lines)}

# Filter Data
line_dict = {idx: line_dict[idx] for idx in line_dict if line_dict[idx][0] != ''}  # remove lines

# Prepare Data
line_dict = {idx: [float(value) for value in line_dict[idx][:-1]] + [line_dict[idx][-1]] for idx in line_dict}
print(line_dict)


# Function: distance of x and y
def distance(x, y):
    return pow(sum([abs(x[i] - y[i]) for i in range(len(x) - 1)]), 0.5)


# Working Data; Select features
...
...
...

# Calc Distance-Tables, sort them and do a first analysis on predictions
distance_dict = {idx: {compare_idx: distance(line_dict[idx], line_dict[compare_idx]) for compare_idx in line_dict
                       if idx != compare_idx} for idx in line_dict}
print(distance_dict)
...
...
...
...

# Show predictions for nearest neighbours
...
...

# Show nodes with different nearest neighbours
...
...
...

# Close up
...

