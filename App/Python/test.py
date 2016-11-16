import hpelm
import numpy

# hpelm.make_hdf5('../Data/csv/forest_x.csv', '../Data/hdf5/forest_x.h5', delimiter=',')
# hpelm.make_hdf5('../Data/csv/forest_t.csv', '../Data/hdf5/forest_t.h5', delimiter=',')

model = hpelm.HPELM(54, 1)
model.add_neurons(2000, 'sigm')
model.add_neurons(10, 'lin')

print 'Training:'
model.train('../Data/hdf5/forest_x.h5', '../Data/hdf5/forest_t.h5')
print 'Predicting:'
model.predict('../Data/hdf5/forest_x.h5', '../Results/forest.h5')

print 'Training error: {}'.format(model.error('../Data/hdf5/forest_t.h5', '../Results/forest.h5'))
