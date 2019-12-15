import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ca2.knn_particular_analysis.knn_particular_analysis import ParticularKnnAnalyst
from ca2.params import *


class IrisAnalyst:
    X = None
    X_train = None
    X_test = None
    y = None
    y_train = None
    y_test = None
    label_map = None
    number_of_records = None
    number_of_records_after_cleaning = None
    particular_knn_analyst = ParticularKnnAnalyst(dnnrange)
    kmp_table = None
    kmp_table_cleaned = None
    knn_accuracy = 0
    svm_accuracy = 0
    rf_accuracy = 0
    ensemble_vote_accuracy = 0
    ensemble_stacking_accuracy = 0

    def analyze(self):
        """
        analyzes the data with different classifiers
        :return:
        """
        self.load_data()

        # kmp analysis
        self.particular_knn_analyst.analyze(self.X, self.y, self.label_map)
        self.kmp_table = self.particular_knn_analyst.get_kmp_table()
        self.clean_data_with_kmp_table()
        self.number_of_records_after_cleaning = self.X.shape[0]
        self.particular_knn_analyst.analyze(self.X, self.y, self.label_map)
        self.kmp_table_cleaned = self.particular_knn_analyst.get_kmp_table()

        # classifier
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=test_size)
        knn_classifier = KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm)
        self.knn_accuracy = self.apply_classifier(knn_classifier)
        svm_classifier = SVC(kernel=kernel, gamma=gamma, C=C)
        self.svm_accuracy = self.apply_classifier(svm_classifier)
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.rf_accuracy = self.apply_classifier(rf_classifier)
        self.ensemble_vote_accuracy = self.apply_classifier(VotingClassifier(
            [('knn', knn_classifier), ('svm', svm_classifier), ('rf', rf_classifier)]))

        # write results
        self.write_file()

    def load_data(self):
        """
        load the data, extract features and labels and split into training and test data
        :return:
        """
        # read all lines, extract header and filter lines
        lines = [line.split(inputseparator) for idx, line in
                 enumerate(open(inputfilename, 'r').read().split('\n'))
                 if line != '']
        feature_col_indices = [idx for idx, feature in enumerate(lines[0][:-1]) if feature in feature_names]
        lines = [line for idx, line in enumerate(lines[1:]) if eval(linefilter)]

        # label set
        label_indices_set = list(set([line[-1] for line in lines]))
        self.label_map = {idx: label for idx, label in enumerate(label_indices_set)}

        # filter features and append calc_features
        data = []
        for line in lines:
            float_line = [float(e) for e in line[:-1]]
            calc_feature_values = [calc_feature(float_line) for calc_feature in calc_features] if addfeatures else None
            filtered_line = [e for idx, e in enumerate(float_line) if idx in feature_col_indices]
            data.append(filtered_line + (calc_feature_values if calc_feature_values is not None else []))

        # save prepared data
        with open(outputfilename, 'w') as prepared_data_file:
            for line in data:
                prepared_data_file.write('{}\n'.format(str(line)))

        # transform to numpy arrays
        self.X = np.array([np.array(line) for line in data])
        self.y = np.array([label_indices_set.index(line[-1]) for line in lines])
        self.number_of_records = len(self.y)

    def clean_data_with_kmp_table(self):
        """
        cleans the data using the results of kmp analysis
        :return:
        """
        for kmp_value in self.kmp_table:
            if kmp_value in dnnrange:
                self.X = np.delete(self.X, self.kmp_table[kmp_value], axis=0)
                self.y = np.delete(self.y, self.kmp_table[kmp_value], axis=0)

    def apply_classifier(self, classifier):
        """
        trains a given classifier and test with test data
        :param classifier: sklearn estimator object
        :return: accuracy score
        """
        classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

    def write_file(self):
        """
        write output file
        :return:
        """
        # header
        lines = []
        lines.append('[Analysis of {}]'.format(inputfilename))
        lines.append('Number of records: {}\n'.format(self.number_of_records))
        OUTPUT_FILE_LINE_BREAK = '\n'

        # kmp analysis
        lines.append(printseparator('KMP analysis'))
        for kmp_value in self.kmp_table:
            records = self.kmp_table[kmp_value]
            lines.append('KMP {}: {} {}'.format(kmp_value, len(records), records))
        lines.append(OUTPUT_FILE_LINE_BREAK)

        # kmp analysis after cleaning
        lines.append(printseparator('KMP analysis after cleaning'))
        for kmp_value in self.kmp_table_cleaned:
            records = self.kmp_table_cleaned[kmp_value]
            lines.append('KMP {}: {} {}'.format(kmp_value, len(records), records))
        lines.append(OUTPUT_FILE_LINE_BREAK)
        lines.append('Number of records after cleaning: {}'.format(self.number_of_records_after_cleaning))
        lines.append(OUTPUT_FILE_LINE_BREAK)

        # classifier accuracy scores
        lines.append(printseparator('Classifier accuracy scores'))
        lines.append('k nearest neighbour: {}%'.format(int(round(self.knn_accuracy * 100))))
        lines.append('support vector machine: {}%'.format(int(round(self.svm_accuracy * 100))))
        lines.append('random forest: {}%'.format(int(round(self.rf_accuracy * 100))))
        lines.append('ensemble learning vote: {}%'.format(int(round(self.ensemble_vote_accuracy * 100))))

        # write lines to file
        with open(data_analysis_report, 'w') as file:
            for line in lines:
                file.write(line)
                if line != OUTPUT_FILE_LINE_BREAK:
                    file.write(OUTPUT_FILE_LINE_BREAK)
            file.close()


if __name__ == '__main__':
    analyst = IrisAnalyst()
    analyst.analyze()
