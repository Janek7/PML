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


class ParticularKnnAnalyst:
    lines = None
    distance_table = None

    def __init__(self, kmp_range):
        self.kmp_range = kmp_range

    def analyze(self, X, y, label_map):
        """
        runs a particular knn analysis
        :param X: feature data
        :param y: label data
        :param label_map: map of label indices and names
        :return:
        """
        # Prepare Data
        self.lines = [{KEY_INDEX: idx,
                       KEY_FEATURES: [e for e in data],
                       KEY_LABEL: label_map[y[idx]]}
                      for idx, data in enumerate(X)]

        # Working Data; Select features
        for line in self.lines:
            line[KEY_FEATURES] = [feature_value for idx, feature_value in enumerate(line[KEY_FEATURES]) if
                                  idx in featurecols]

        # Calc Distance-Tables
        self.distance_table = [{KEY_INDEX: single_line_dict[KEY_INDEX],
                                KEY_LABEL: single_line_dict[KEY_LABEL],
                                KEY_FEATURES: single_line_dict[KEY_FEATURES],
                                KEY_NEIGHBOURS:
                                    [{KEY_INDEX: neighbour_line_dict[KEY_INDEX],
                                      KEY_DISTANCE: self.euclidean_distance(single_line_dict[KEY_FEATURES],
                                                                            neighbour_line_dict[KEY_FEATURES]),
                                      KEY_LABEL: neighbour_line_dict[KEY_LABEL]}
                                     for neighbour_line_dict in self.lines if
                                     single_line_dict[KEY_INDEX] != neighbour_line_dict[KEY_INDEX]]}
                               for single_line_dict in self.lines]

        # sort and extend table
        for line in self.distance_table:
            line[KEY_NEIGHBOURS].sort(key=lambda x: x[KEY_DISTANCE])
            line[KEY_KMP] = self.get_kmp(line)
            line[KEY_10_NEIGHBOURS] = self.get_neighbour_values(line)

    def get_kmp_table(self):
        """
        returns a table of kmp values and nodes with this value
        :return:
        """
        return {kmp_value: [x[KEY_INDEX] for x in list(filter(lambda x: x[KEY_KMP] == kmp_value, self.distance_table))]
                for kmp_value in self.kmp_range}

    @staticmethod
    def euclidean_distance(x, y):
        """
        computes the euclidean distance between two vectors
        :param x: vector a
        :param y: vector b
        :return: distance
        """
        return pow(sum([abs(x[i] - y[i]) for i in range(len(x))]), 0.5)

    @staticmethod
    def get_kmp(line):
        """
        computes the kmp number
        :param line: data record
        :return: kmp
        """
        kmp = 0
        for neighbour in line[KEY_NEIGHBOURS]:  # so lange bis ein neighbour ein anderes label hat
            if neighbour[KEY_LABEL] == line[KEY_LABEL]:
                kmp += 1
            else:
                break
        return kmp

    @staticmethod
    def get_neighbour_values(line):
        """
        Calculate values (?) for x nearest neighbours
        :return:
        """
        list = []
        sum = 0
        for idx, neighbour in enumerate(line[KEY_NEIGHBOURS][:firstrelpred]):
            sum += (1 if line[KEY_LABEL] == neighbour[KEY_LABEL] else 0)
            value = (sum / (idx + 1))
            list.append((idx + 1, round(value, rpdigits)))
        return list

    def write_file(self, path):
        """
        writes all results into a result file
        :param path: path of result file
        :return:
        """
        file = open(path, 'w')

        file.write('Particular Data Analysis (with kNN) [1.2]\n\n')
        file.write('Selected features: {}\n'.format(featurecols))
        file.write('Ignored lines: {}\n'.format(linefilter))
        file.write(printseparator + '\n')
        file.write('count of data records: {}\n\n'.format(len(self.lines)))

        for line in self.distance_table:
            file.write(
                '{}: {} - {} - KMP: {} - {}\n'.format(line[KEY_INDEX], line[KEY_FEATURES], line[KEY_LABEL],
                                                      line[KEY_KMP],
                                                      line[KEY_10_NEIGHBOURS]))

        file.write(printseparator + '\n')
        file.write('Nodes with different first neighbours\n')
        kmp_table = self.get_kmp_table()
        for kmp_value in kmp_table:
            nodes = kmp_table[kmp_value]
            file.write('KMP {}: {} {}\n'.format(kmp_value, len(nodes), nodes))

        file.close()


if __name__ == '__main__':
    analyst = ParticularKnnAnalyst(dnnrange)
    # analyst.load_data()
    # analyst.write_file('iris.data.erg')
