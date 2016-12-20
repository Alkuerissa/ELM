import preprocessor
import hpelm
import error
import time
import h5py
import os
import numpy as np


class Benchmark:

    def __init__(self, name, training_percentage, neurons):
        self.name = name
        self.training_percentage = training_percentage
        self.neurons = neurons
        self.data_path = None
        self.results_path = None
        self.output_path = None
        self.data = None
        self.results = None
        self.training_data = None
        self.test_data = None
        self.training_results = None
        self.test_results = None

    @staticmethod
    def small_benchmark(name, training_percentage, neurons, data, results):
        obj = Benchmark(name, training_percentage, neurons)
        obj.data = data
        if obj.data.shape.__len__() == 1:
            obj.data = obj.data.reshape(obj.data.shape[0], 1)
        obj.results = results
        if obj.results.shape.__len__() == 1:
            obj.results = obj.results.reshape(obj.results.shape[0], 1)
        return obj

    @staticmethod
    def big_benchmark(name, training_percentage, neurons, data_path, results_path, output_path):
        obj = Benchmark(name, training_percentage, neurons)
        obj.data_path = data_path
        obj.results_path = results_path
        obj.output_path = output_path
        return obj

    def run(self):
        if self.data_path is None and self.data is not None:
            self.data = preprocessor.normalize(self.data)
            self.training_data, self.test_data, self.training_results, self.test_results = \
                preprocessor.split(self.data, self.results, self.training_percentage)
            errors = []
            percentages = []
            times = []
            for i in range(0, self.neurons.__len__()):
                model = hpelm.ELM(self.data.shape[1], self.results.shape[1], classification='c')
                for j in range(0, self.neurons[i].__len__()):
                    model.add_neurons(self.neurons[i][j][0], self.neurons[i][j][1])
                start = time.time()
                model.train(self.training_data, self.training_results)
                end = time.time()
                times.append(end - start)
                res = model.predict(self.test_data)
                errors.append(model.error(self.test_results, res))
                percentages.append(error.percentage(self.test_results, res))

            return errors, percentages, times
        if self.data_path is not None and self.data is None:
            if self.output_path[-1] != '/':
                self.output_path += '/'
            self.output_path += self.name + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            preprocessor.normalize_hdf5(self.data_path, self.output_path + self.name + '_normalized.h5')
            self.data_path = self.output_path + self.name + '_normalized.h5'
            preprocessor.split_hdf5(self.data_path, self.results_path, self.training_percentage,
                                    self.output_path + self.name + '_training_data.h5',
                                    self.output_path + self.name + '_test_data.h5',
                                    self.output_path + self.name + '_training_results.h5',
                                    self.output_path + self.name + '_test_expected.h5')
            errors = []
            percentages = []
            times = []
            data = h5py.File(self.data_path, 'r')
            n = data['data'].shape[1]
            data.close()
            results = h5py.File(self.results_path, 'r')
            o = results['data'].shape[1]
            results.close()
            for i in range(0, self.neurons.__len__()):
                model = hpelm.HPELM(n, o, classification='c')
                for j in range(0, self.neurons[i].__len__()):
                    model.add_neurons(self.neurons[i][j][0], self.neurons[i][j][1])
                start = time.time()
                model.train(self.output_path + self.name + '_training_data.h5', self.output_path + self.name + '_training_results.h5')
                end = time.time()
                times.append(end - start)
                model.predict(self.output_path + self.name + '_test_data.h5', self.output_path + self.name + '_test_output.h5')
                errors.append(model.error(self.output_path + self.name + '_test_expected.h5', self.output_path + self.name + '_test_output.h5'))
                percentages.append(error.hdf5_percentage(self.output_path + self.name + '_test_expected.h5', self.output_path + self.name + '_test_output.h5'))
            return errors, percentages, times
        return [], [], []
