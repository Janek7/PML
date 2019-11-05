# dict keys
KEY_INDEX = 'idx'
KEY_FEATURES = 'features'
KEY_NEIGHBOURS = 'neighbours'
KEY_DISTANCE = 'distance'
KEY_LABEL = 'label'
KEY_KMP = 'kmp'
KEY_10_NEIGHBOURS = '10_neighbours'

# Params
inputfilename = 'iris.data'  # filename input data
outputfilename = inputfilename + '.erg'  # filename output data
inputseparator = ','  # separator for csv columns
labelcolumn = 4  # column of label
columnfilter = [0, 1, 2, 3]  # columns with features
columnflen = len(columnfilter)  # count of features in data
featurecols = [0, 1, 2, 3]  # selected Features
linefilter = [0]  # linefilter: lines to ignore
firstrelpred = 10  # count of first neighbours in outputlist
rpdigits = 4  # relative prediction: number of digits after decimal point
dnnrange = [0, 1, 2, 3, 4]  # show different nearest neighbour range
sepaprod = 'lambda x,y: x*y'
printseparator = '*' * 80 + '\n'

file = open(outputfilename, 'w')

# Header in Outputfile
file.write('Particular Data Analysis (with kNN) [1.2]\n\n')
file.write('Selected features: {}\n'.format(featurecols))
file.write('Ignored lines: {}\n'.format(linefilter))
file.write(printseparator + '\n')

# Load Data
lines = [line.split(inputseparator) for line in open(inputfilename, 'r').read().split('\n')]
lines = {idx: line for idx, line in enumerate(lines) if line[0] != '' and idx not in linefilter}
file.write('count of data records: {}\n\n'.format(len(lines)))

# Prepare Data
lines = [{KEY_INDEX: idx,
          KEY_FEATURES: [float(value) for value in lines[idx][:-1]],
          KEY_LABEL: lines[idx][-1]}
         for idx in lines]


# Function: distance of x and y
def distance(x, y):
    return pow(sum([abs(x[i] - y[i]) for i in range(len(x))]), 0.5)


# Working Data; Select features
for line in lines:
    line[KEY_FEATURES] = [feature_value for idx, feature_value in enumerate(line[KEY_FEATURES]) if idx in featurecols]

# Calc Distance-Tables
distance_table = [{KEY_INDEX: single_line_dict[KEY_INDEX],
                   KEY_LABEL: single_line_dict[KEY_LABEL],
                   KEY_FEATURES: single_line_dict[KEY_FEATURES],
                   KEY_NEIGHBOURS:
                       [{KEY_INDEX: neighbour_line_dict[KEY_INDEX],
                         KEY_DISTANCE: distance(single_line_dict[KEY_FEATURES], neighbour_line_dict[KEY_FEATURES]),
                         KEY_LABEL: neighbour_line_dict[KEY_LABEL]}
                        for neighbour_line_dict in lines if
                        single_line_dict[KEY_INDEX] != neighbour_line_dict[KEY_INDEX]]}
                  for single_line_dict in lines]


# Calculate KMP value
def get_kmp(line):
    kmp = 0
    for neighbour in line[KEY_NEIGHBOURS]:  # so lange bis ein neighbour ein anderes label hat
        if neighbour[KEY_LABEL] == line[KEY_LABEL]:
            kmp += 1
        else:
            break
    return kmp


# Calculate values (?) for 10 nearest neighbours
def get_neighbour_values(line):
    list = []
    sum = 0
    for idx, neighbour in enumerate(line[KEY_NEIGHBOURS][:firstrelpred]):
        sum += (1 if line[KEY_LABEL] == neighbour[KEY_LABEL] else 0)
        value = (sum / (idx + 1))
        list.append((idx + 1, round(value, rpdigits)))
    return list


# sort and extend table
for line in distance_table:
    line[KEY_NEIGHBOURS].sort(key=lambda x: x[KEY_DISTANCE])
    line[KEY_KMP] = get_kmp(line)
    line[KEY_10_NEIGHBOURS] = get_neighbour_values(line)
    # file.write(line[KEY_INDEX], line[KEY_LABEL], line[KEY_KMP])

# Show predictions for nearest neighbours
for line in distance_table:
    file.write(
        '{}: {} - {} - KMP: {} - {}\n'.format(line[KEY_INDEX], line[KEY_FEATURES], line[KEY_LABEL], line[KEY_KMP],
                                              line[KEY_10_NEIGHBOURS]))

# Show nodes with different nearest neighbours
file.write(printseparator + '\n')
file.write('Nodes with different first neighbours\n')
for kmp_value in dnnrange:
    nodes = [x[KEY_INDEX] for x in list(filter(lambda x: x[KEY_KMP] == kmp_value, distance_table))]
    file.write('KMP {}: {} {}\n'.format(kmp_value, len(nodes), nodes))

# Close up
file.close()
