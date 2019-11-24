# Parameter for data selection
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  #
calc_features = []  # [lambda x: x[0] * 2]
line_pre_filter = 'idx not in []'
k_max_predict = [0]

# Parameter for file handling
input_file = 'iris_with_header.data'
output_file = 'ca2.output'
csv_separator = ','
csv_line_separator = '\n'
output_file_line_break = '\n'
section_header = lambda x: (' ' + x + ' ').center(50, '-')
save_fig = False

# Paramter for knn classification
k = 5
weights = 'uniform'  # 'uniform', 'distance'
trainig_size = 0.67
algorithm = 'auto'  # 'auto', 'ball_tree', 'kd_tree', 'brute'

# Parameter for svm classification
kernel = 'linear'  # 'linear', 'rbf', 'poly'
gamma = 0.7
degree = 0.3
C = 1.0

# Parameter for rf classification
n_estimators = 3
random_state = None

# Parameter for particular knn analysis
kmp_range = [0, 1, 2, 3, 4]

# Parameter for ensemble classification
voting = 'hard'  # 'hard' ofr 'soft'
use_probas = True
