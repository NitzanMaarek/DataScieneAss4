import os
import re
from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

class GUI:

    def __init__(self):
        self.root = Tk()
        self.root.title('Naive Bayes Classifier')
        self.root.geometry('610x200')
        self.frame = Frame(self.root)
        self.frame.pack(side=TOP)

        self.dir_path_label = Label(self.frame, text='Directory Path')
        self.discretization_label = Label(self.frame, text='Discretization Bins')

        self.dir_path_entry = Entry(self.frame, width=50)
        self.discretization_entry = Entry(self.frame, width=50)

        self.browse_button = Button(self.frame, text='Browse', command=self.browse_input_dir, width=10)
        self.build_button = Button(self.frame, text='Build', command=self.build_dataset, width=30)
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
        if self._check_if_files_exist(dir_name) and self._check_bins_input():
            attributes_to_values_dict = analyze_and_get_structure_dictionary(dir_name)
            classifier = Classifier(dir_name, attributes_to_values_dict, number_of_bins)


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

    def classify_dataset(self):
        print('bruuhh')


class Classifier:

    def __init__(self, dir_name, attributes_dictionary, number_of_bins):
        self.dir_name = dir_name
        self.attributes_to_values_dict = attributes_dictionary
        self.number_of_bins = int(number_of_bins)
        self.data = pd.read_csv(dir_name + '\\train.csv')
        self._prepare_data()

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
