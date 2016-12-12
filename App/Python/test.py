# coding=utf-8
import benchmark as bn
import hpelm
from Tkinter import *
from tkFileDialog import askopenfilename

# b = bn.Benchmark.big_benchmark('forest_test', 0.8, [[[54, 'lin'], [500, 'sigm']],
#                                                     [[54, 'lin'], [500, 'tanh']],
#                                                     [[54, 'lin'], [250, 'sigm'], [250, 'tanh']],
#                                                     [[54, 'lin'], [1000, 'sigm'], [1000, 'tanh']]],
#                                '../Data/hdf5/forest_x_not_normalized.h5',
#                                '../Data/hdf5/forest_t.h5',
#                                '../Results/')
# errors, percentages, times = b.run()
# print 'Error: {}'.format(errors)
# print 'Correct percentage: {}'.format(percentages)
# print 'Time: {}'.format(times)

training_path = ""
results_path = ""
h5 = None

root = Tk()
root.title("ELM")


def file_select():
    global h5
    global training_path
    global results_path
    options = {'parent': root, 'title': 'Input data',
               'filetypes': [('csv and hdf5 files', ('*.csv', '*.h5'))]}
    training_path = askopenfilename(**options)
    if training_path == "":
        return
    if training_path[-3:] == '.h5':
        change_h5(True)
    else:
        change_h5(False)
    if h5:
        options['filetypes'] = [('hdf5 files', '.h5')]
    else:
        options['filetypes'] = [('csv files', '.csv')]
    options['title'] = 'Expected results'
    results_path = askopenfilename(**options)
    if results_path == "":
        change_h5(False)
        training_path = ""


def convert():
    hpelm.make_hdf5(training_path, training_path[:-4] + ".h5")
    hpelm.make_hdf5(results_path, results_path[:-4] + ".h5")


def change_h5(val):
    global h5
    h5 = val
    if val == True:
        convert_button.config(state='disabled')
    elif val == False:
        convert_button.config(state='normal')

file_select_button = Button(root, text="Select data", command=file_select)
file_select_button.pack()
convert_button = Button(root, text="Convert to hdf5", state='disabled', command=convert)
convert_button.pack()

root.mainloop()

