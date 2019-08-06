import os
import re
from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

YES = 'Y'
NO = 'N'
M = 2

class GUI:

    def __init__(self):
        self.root = Tk()
        self.root.title('Naive Bayes Classifier')
        self.root.geometry('610x200')
        self.frame = Frame(self.root)
        self.frame.pack(side=TOP)

        self.dir_path_label = Label(self.frame, text='Directory Path')
        self.discretization_label = Label(self.frame, text='Discretization Bins')

        self.dir_path_string_var = StringVar(self.frame)
        self.discretization_string_var = StringVar(self.frame)

        self.dir_path_string_var.trace("w", self._check_fields)
        self.discretization_string_var.trace("w", self._check_fields)

        self.dir_path_entry = Entry(self.frame, width=50, textvariable=self.dir_path_string_var)
        self.discretization_entry = Entry(self.frame, width=50, textvariable=self.discretization_string_var)

        self.browse_button = Button(self.frame, text='Browse', command=self.browse_input_dir, width=10)
        self.build_button = Button(self.frame, text='Build', command=self.build_dataset, width=30)
        self.build_button.config(state='disabled')
        self.classify_button = Button(self.frame, text='Classify', command=self.classify_dataset, width=30)

        self.dir_path_label.grid(row=0, column=0)
        self.dir_path_entry.grid(row=0, column=1)
        self.browse_button.grid(row=0, column=2)

        self.discretization_label.grid(row=1, column=0)
        self.discretization_entry.grid(row=1, column=1)

        self.build_button.grid(row=2, column=1)
        self.classify_button.grid(row=3, column=1)

        self.root.mainloop()

    def browse_input_dir(self):
        curr_directory = os.getcwd()
        dir_name = filedialog.askdirectory(initialdir=curr_directory, title="Select a directory")
        if len(dir_name) > 0:
            if not os.path.isdir(dir_name):
                messagebox.showerror('Naive Bayes Classifier', 'Please select a directory')
            else:
                self.dir_path_entry.delete(0, END)
                self.dir_path_entry.insert(INSERT, dir_name)
        else: # I doubt it will show an error message because you must select a directory or cancel the browse process.
            messagebox.showerror('Naive Bayes Classifier', 'Please select a directory')

    def build_dataset(self):
        dir_name = self.dir_path_entry.get()
        number_of_bins = self.discretization_entry.get()
        # if self._check_if_files_exist(dir_name) and self._check_bins_input():
        attributes_to_values_dict = analyze_and_get_structure_dictionary(dir_name)
        self.classifier = Classifier(dir_name, attributes_to_values_dict, number_of_bins)
        self.classifier.fit()

    def _check_fields(self, *args):
        dir_name = self.dir_path_entry.get()
        if self._check_if_files_exist(dir_name) and self._check_bins_input():
            self.build_button.config(state='normal')
        else:
            self.build_button.config(state='disabled')


    def _check_bins_input(self):
        number_of_bins = self.discretization_entry.get()
        pattern = re.compile("[1-9]([0-9])*")
        if pattern.match(number_of_bins):
            return True
        else:
            messagebox.showerror('Naive Bayes Classifier', 'Please enter a valid number of bins')
            return False

    def _check_if_files_exist(self, dir_name):
        if not os.path.isdir(dir_name):
            messagebox.showerror('Naive Bayes Classifier', 'Please select a directory')
            return False
        if not os.path.isfile(dir_name + '\\Structure.txt'):
            messagebox.showerror('Naive Bayes Classifier', 'Structure.txt file is missing')
            return False
        if not os.path.isfile(dir_name + '\\train.csv'):
            messagebox.showerror('Naive Bayes Classifier', 'train.csv file is missing')
            return False
        if not os.path.isfile(dir_name + '\\test.csv'):
            messagebox.showerror('Naive Bayes Classifier', 'test.csv file is missing')
            return False
        return True

    def _write_output(self, train_with_prediction):
        train_with_prediction['output_index'] = [i + 1 for i in train_with_prediction.index]
        output = train_with_prediction[['output_index', 'prediction']]
        output_path = self.dir_path_entry.get() + '\\output.txt'
        output.to_csv(output_path, header=None, index=None, sep=' ')

    def classify_dataset(self):
        train_path = self.dir_path_entry.get() + '\\test.csv'
        train = pd.read_csv(train_path, index_col=False)
        train_with_prediction = self.classifier.predict(train)
        self._write_output(train_with_prediction)

class Classifier:

    def __init__(self, dir_name, attributes_dictionary, number_of_bins):
        self.dir_name = dir_name
        self.attributes_to_values_dict = attributes_dictionary
        self.number_of_bins = int(number_of_bins)
        self.data = pd.read_csv(dir_name + '\\train.csv')
        self._prepare_data()
        self.n = len(self.data)
        self.m = 2

    def _prepare_data(self):
        for key in self.attributes_to_values_dict:
            if self.attributes_to_values_dict[key] == 'NUMERIC':
                self._fill_empty_cells_of_numeric(key)
            else:
                self._fill_empty_cells_of_category(key)

    def _fill_empty_cells_of_category(self, attribute_name):
        most_common_value = self.data[attribute_name].value_counts().argmax()
        self.data[attribute_name].fillna(most_common_value, inplace=True)   
        print(self.data)

    def _fill_empty_cells_of_numeric(self, attribute_name):
        mean_value = self.data[attribute_name].mean()
        self.data[attribute_name].fillna(mean_value, inplace=True)
        self._bin_numeric_column(attribute_name)

    def _bin_numeric_column(self, attribute_name):
        max_value = self.data[attribute_name].max()
        min_value = self.data[attribute_name].min()
        w = (max_value - min_value)/self.number_of_bins
        bins = []
        for i in range(0, self.number_of_bins):
            interval = min_value + i*w
            bins.append(interval)
        bins.append(max_value)
        self.data[attribute_name + '_binned'] = pd.cut(self.data[attribute_name], 3)
        print(self.data) #testing binning

    # # TODO: need to count for 1 and for 0?
    # def _calculate_probabilites(self):
    #     self.probabilities_dictionary = {}
    #     self.num_of_rows = len(self.data)
    #
    #     for column in self.data.columns:
    #         counts = self.data[column].value_counts()
    #         self.probabilities_dictionary[column] = {}
    #         for value in counts:
    #             self.probabilities_dictionary[column][value] = counts[value] / self.num_of_rows


    def fit(self):
        """
        Using m-estimate, m=2
        :return:
        """
        self.value_counts = {}
        label = 'class'
        label_column = self.data[label]
        # labels_num = label_column.nunique()
        # labels = label_column.unique()

        self.samples_count = len(self.data)
        self.yes_count = len(self.data[self.data['class'] == 'Y'])
        self.no_count = len(self.data[self.data['class'] == 'N'])

        for column in self.data.columns:
            if column == label:
                continue
            self.value_counts[column] = {}
            distribution = self.data.groupby([column, label_column]).size().reset_index().rename(columns={0: 'count'})
            for i, row in distribution.iterrows():
                column_value = row[column]
                label_value = row[label]
                if column_value not in self.value_counts[column]:
                    self.value_counts[column][column_value] = {}
                self.value_counts[column][column_value][label_value] = row['count']

    def _predict_row_label(self, row):
        possible_predictions = ['N', 'Y']
        max_proba = 0
        prediction = None
        for p_prediction in possible_predictions:
            proba = self._calculate_probability(row, p_prediction)
            if proba > max_proba:
                prediction = p_prediction
                max_proba = proba
        return prediction, max_proba

    def _calculate_probability(self, row, p_prediction):
        probabilities = []
        for tup in row.items():
            if tup[0] == 'class':
                continue
            if tup[0] in self.value_counts:
                if tup[1] in self.value_counts[tup[0]]:
                    if p_prediction in self.value_counts[tup[0]][tup[1]]:
                        nc = self.value_counts[tup[0]][tup[1]][p_prediction]
            else:
                nc = 0
            if p_prediction == 'Y':
                n = self.yes_count
            else:
                n = self.no_count
            probabilities.append(self.m_estimate(nc, n, M))

        result = 1
        for p in probabilities:
            result = result * p
        print result * float(self.samples_count) / n
        return result * float(self.samples_count) / n

    @staticmethod
    def m_estimate(nc, n, m):
        p = float(1) / m
        return float((nc + m * p) / (n + m))


    def predict(self, train):
        predictions = []
        probabilities = []
        for i, row in train.iterrows():
            prediction, proba = self._predict_row_label(row)
            predictions.append(prediction)
            probabilities.append(proba)
        train['prediction'] = predictions
        train['probability'] = probabilities
        return train


def analyze_and_get_structure_dictionary(dir_name):
    structure_dictionary_attribute_to_values = {}
    structure_file = open(dir_name + '\\Structure.txt', 'r')
    for line in structure_file:
        splitted_line = line.split()
        attribute_name = splitted_line[1]
        if splitted_line[2] == 'NUMERIC':
            structure_dictionary_attribute_to_values[attribute_name] = 'NUMERIC'
        else:
            splitted_line[2] = ' '.join(splitted_line[2:])
            values = splitted_line[2]
            splitted_line = values[1:-1]
            splitted_values = splitted_line.split(',')
            structure_dictionary_attribute_to_values[attribute_name] = splitted_values
    structure_file.close()
    return structure_dictionary_attribute_to_values

if __name__ == "__main__":
    gui = GUI()
