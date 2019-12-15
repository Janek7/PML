# params
inputfilename = 'iris.data'  # filename input data
outputdir = 'output\\'
outputfilename = outputdir + inputfilename + '.erg'  # filename output data
data_analysis_report = outputdir + inputfilename + '.dar'  # filename report data analysis
# ensemble_learning_report = inputfilename + '.elr'               #filename report ensemble learning

addfeatures = True  # allow new features
calc_features = [lambda x: x[0] * 2]  # new feature
feature_names = ['petal_length', 'petal_width', 'calc_feature', 'iris_type']

firstrelpred = 10  # count of first neighbours in outputlist
dnnrange = [0, 1, 2, 3, 4]  # show different nearest neighbour range
max_KMP = [0]

inputseparator = ';'  # separator for csv columns
linefilter = 'idx not in []'  # linefilter: lines to ignore
filterfunction = "j % 2 == 0"  # für was?
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
random_state = None

# Parameter for ensemble classification
voting = 'hard'  # 'hard' ofr 'soft'
use_probas = True
