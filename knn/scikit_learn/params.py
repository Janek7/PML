# Paramter for knn classification
k = 5
weights = 'uniform'  # 'uniform', 'distance'
trainig_size = 0.67
algorithm = 'auto'  # 'auto', 'ball_tree', 'kd_tree', 'brute'

# Parameter for data selection
feature_cols = ['sepal_length', 'sepal_width', 'petal_length',
                'petal_width']  # 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
calc_features = [lambda x: x[0] * 2]
line_pre_filter = 'idx not in []'
k_max_predict = 0

# Parameter for data input
input_data_file = 'iris_with_header.data'
csv_separator = ','
csv_line_separator = '\n'
