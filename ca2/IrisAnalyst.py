import numpy as np
from sklearn.model_selection import train_test_split
from ca2.ca2_params import *
from ca2.classifiers.ensemble_wrappers import VotingClassifierWrapper
from ca2.classifiers.k_nearest_neighbour import KnnClassifierWrapper
from ca2.classifiers.random_forest import RfClassifierWrapper
from ca2.classifiers.support_vector_machine import SvmClassifierWrapper
from ca2.knn_particular_analysis.knn_particular_analysis import ParticularKnnAnalyst


class IrisAnalyst:
    X = None
    y = None
    label_map = None
    number_of_records = None
    number_of_records_after_cleaning = None
    particular_knn_analyst = ParticularKnnAnalyst(kmp_range)
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
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1 - trainig_size)

        knn_classifier = KnnClassifierWrapper()
        knn_classifier.train(X_train, y_train)
        self.knn_accuracy = knn_classifier.validate(X_test, y_test)
        # knn_classifier.plot(save_fig)

        svm_classifier = SvmClassifierWrapper()
        svm_classifier.train(X_train, y_train)
        self.svm_accuracy = svm_classifier.validate(X_test, y_test)
        # svm_classifier.plot(save_fig)

        rf_classifier = RfClassifierWrapper()
        rf_classifier.train(X_train, y_train)
        self.rf_accuracy = rf_classifier.validate(X_test, y_test)
        # rf_classifier.plot2(save_fig)

        vote_classifier = VotingClassifierWrapper(
            [('knn', knn_classifier.classifier), ('svm', svm_classifier.classifier), ('rf', rf_classifier.classifier)])
        vote_classifier.train(X_train, y_train)
        self.ensemble_vote_accuracy = rf_classifier.validate(X_test, y_test)

        # write results
        self.write_file()

    def load_data(self):
        """
        load the data, extract features and labels and split into training and test data
        :return:
        """
        # read all lines, extract header and filter lines
        lines = [line.split(csv_separator) for idx, line in
                 enumerate(open(input_file, 'r').read().split(csv_line_separator))
                 if line != '']
        feature_col_indices = [idx for idx, feature in enumerate(lines[0][:-1]) if feature in feature_cols]
        lines = [line for idx, line in enumerate(lines[1:]) if eval(line_pre_filter)]

        # label set
        label_indices_set = list(set([line[-1] for line in lines]))
        self.label_map = {idx: label for idx, label in enumerate(label_indices_set)}

        # filter features and append calc_features
        data = []
        for line in lines:
            new_line = [float(e) for idx, e in enumerate(line) if idx in feature_col_indices]
            for calc_feature in calc_features:  # Todo: calc features davor berechnen
                new_line.append(calc_feature(new_line))
            data.append(new_line)

        # save prepared data
        with open(output_filename_prepared_data, 'w') as prepared_data_file:
            for line in lines:
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
            if kmp_value in k_max_predict:
                self.X = np.delete(self.X, self.kmp_table[kmp_value], axis=0)
                self.y = np.delete(self.y, self.kmp_table[kmp_value], axis=0)

    def write_file(self):
        """
        write output file
        :return:
        """
        # header
        lines = []
        lines.append('[Analysis of {}]'.format(input_file))
        lines.append('Number of records: {}'.format(self.number_of_records))
        lines.append(output_file_line_break)

        # kmp analysis
        lines.append(section_header('KMP analysis'))
        for kmp_value in self.kmp_table:
            records = self.kmp_table[kmp_value]
            lines.append('KMP {}: {} {}'.format(kmp_value, len(records), records))
        lines.append(output_file_line_break)

        # kmp analysis after cleaning
        lines.append(section_header('KMP analysis after cleaning'))
        for kmp_value in self.kmp_table_cleaned:
            records = self.kmp_table_cleaned[kmp_value]
            lines.append('KMP {}: {} {}'.format(kmp_value, len(records), records))
        lines.append(output_file_line_break)
        lines.append('Number of records after cleaning: {}'.format(self.number_of_records_after_cleaning))
        lines.append(output_file_line_break)

        # classifier accuracy scores
        lines.append(section_header('Classifier accuracy scores'))
        lines.append('k nearest neighbour: {}%'.format(int(round(self.knn_accuracy * 100))))
        lines.append('support vector machine: {}%'.format(int(round(self.svm_accuracy * 100))))
        lines.append('random forest: {}%'.format(int(round(self.rf_accuracy * 100))))
        lines.append('ensemble learning vote: {}%'.format(int(round(self.ensemble_vote_accuracy * 100))))

        # write lines to file
        with open(output_file, 'w') as file:
            for line in lines:
                file.write(line)
                if line != output_file_line_break:
                    file.write(output_file_line_break)
            file.close()


if __name__ == '__main__':
    analyst = IrisAnalyst()
    analyst.analyze()
