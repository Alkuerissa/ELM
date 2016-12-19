import numpy
import math
import h5py
from sklearn import preprocessing


def open_csv(path):
    return numpy.loadtxt(open(path, "rb"), delimiter=",")


def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler() #(val - min)/(max - min)
    return min_max_scaler.fit_transform(data)


def normalize_hdf5(data_path, output_path):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = h5py.File(data_path, 'r')
    normalized_data = h5py.File(output_path, 'w')
    normalized_data.create_dataset('data', data['data'].shape, float, min_max_scaler.fit_transform(data['data'][:]))
    data.close()
    normalized_data.close()
    return


def split(data, results, training_percentage):
    n1 = int(math.floor(training_percentage * data.shape[0]))
    n2 = int(math.floor(training_percentage * results.shape[0]))
    return data[:n1], data[n1:], results[:n2], results[n2:]


def split_hdf5(data_path, results_path, training_percentage, training_data_path, test_data_path, training_results_path, test_results_path):
    data = h5py.File(data_path, 'r')
    results = h5py.File(results_path, 'r')
    n1 = int(math.floor(training_percentage * data['data'].shape[0]))
    n2 = int(math.floor(training_percentage * results['data'].shape[0]))
    training_data = h5py.File(training_data_path, 'w')
    test_data = h5py.File(test_data_path, 'w')
    training_results = h5py.File(training_results_path, 'w')
    test_results = h5py.File(test_results_path, 'w')
    training_data.create_dataset('data', (n1, data['data'].shape[1]), float, data['data'][:n1])
    training_data.close()
    test_data.create_dataset('data', (data['data'].shape[0] - n1, data['data'].shape[1]), float, data['data'][n1:])
    test_data.close()
    data.close()
    training_results.create_dataset('data', (n2, results['data'].shape[1]), float, results['data'][:n2])
    training_results.close()
    test_results.create_dataset('data', (results['data'].shape[0] - n2, results['data'].shape[1]), float, results['data'][n2:])
    test_results.close()
    results.close()
    return
