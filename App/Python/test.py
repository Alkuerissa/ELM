import hpelm
import numpy
#print 'Making hdf5...'
#hpelm.make_hdf5('../Data/csv/forest_x.csv', '../Data/hdf5/forest_x.h5', delimiter=',')
#hpelm.make_hdf5('../Data/csv/forest_t.csv', '../Data/hdf5/forest_t.h5', delimiter=',')
#print 'Normalizing...'
#hpelm.normalize_hdf5('../Data/hdf5/forest_x.h5')
print 'Creating model...'
model = hpelm.HPELM(54, 1, classification='c', tprint=1)
model.add_neurons(1000, 'tanh')
model.add_neurons(1000, 'sigm')
model.add_neurons(1000, 'lin')
print 'Training:'
model.train('../Data/hdf5/forest_x.h5', '../Data/hdf5/forest_t.h5')
print 'Predicting:'
model.predict('../Data/hdf5/forest_x.h5', '../Results/forest.h5')
print 'Training error: {}'.format(model.error('../Data/hdf5/forest_t.h5', '../Results/forest.h5'))
