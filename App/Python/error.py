import h5py
import numpy


def percentage(expected, actual):
    return numpy.count_nonzero(expected == actual) / (expected.shape[0] * expected.shape[1])


def hdf5_percentage(expected_path, actual_path):
    expected = h5py.File(expected_path, 'r')
    actual = h5py.File(actual_path, 'r')
    result = numpy.count_nonzero(expected['data'][:] == numpy.round(actual['data'][:])) / float(expected['data'].shape[0] * expected['data'].shape[1])
    expected.close()
    actual.close()
    return result
