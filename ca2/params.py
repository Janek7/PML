# params
inputfilename = 'iris0001.csv'  # filename input data
# inputfilename = 'iris.data'  # filename input data
outputdir = 'output\\'
filtereddata = outputdir + inputfilename + '.filtered'
outputfilename = outputdir + inputfilename + '.erg'  # filename output data
data_analysis_report = outputdir + inputfilename + '.dar'  # filename report data analysis
# ensemble_learning_report = inputfilename + '.elr'               #filename report ensemble learning

addfeatures = True  # allow new features
calc_features = [lambda x: x[2] / x[3]]  # new feature
feature_names = ['sepal_length', 'sepal_width']  # 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'

firstrelpred = 10  # count of first neighbours in outputlist
dnnrange = [0, 1, 2, 3, 4]  # show different nearest neighbour range
max_KMP = [0]

inputseparator = ';'  # separator for csv columns
linefilter = 'idx not in []'  # linefilter: lines to ignore
filterfunction = "idx % 2 == 0"  # beispiel: idx % 2 == 0
rpdigits = 4  # relative prediction: number of digits after decimal point
printseparator = lambda x: (' ' + x + ' ').center(50, '-')

test_size = 0.33  # Test Size

# Paramter for knn classification
n_neighbors = 5
weights = 'uniform'  # 'uniform', 'distance'
algorithm = 'auto'  # 'auto', 'ball_tree', 'kd_tree', 'brute'

# Parameter for svm classification
kernel = 'linear'  # 'linear', 'rbf', 'poly'
gamma = 0.7
degree = 0.3
C = 1.0

# Parameter for rf classification
n_estimators = 3
random_state = 34

# Parameter for ensemble classification
voting = 'hard'  # 'hard' ofr 'soft'