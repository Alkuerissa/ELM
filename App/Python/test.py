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
name = "No data loaded!"
h5 = None
benchmark = None
neurons = [[]]
current_benchmark = 0
neuronsnum = 0
functions = ['lin', 'sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf']
function_names = ['Linear', 'Sigmoid', 'Hyperbolic tangent', 'RBF (L1)', 'RBF (L2)', 'RBF (L∞)']


root = Tk()
root.title("ELM")
root.geometry("200x350")
function = StringVar(root, function_names[0])


def file_select():
    global h5
    global training_path
    global results_path
    global name
    options = {'parent': root, 'title': 'Input data',
               'filetypes': [('csv and hdf5 files', ('*.csv', '*.h5'))]}
    training_path = askopenfilename(**options)
    if training_path == "":
        change_h5(None)
        set_name("No data loaded!")
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
        change_h5(None)
        set_name("No data loaded!")
        training_path = ""
    else:
        ind = training_path.rfind('/')
        if ind < 0:
            ind = training_path.rfind('\\')
        if ind < 0:
            ind = 0
        set_name(training_path[ind + 1:training_path.__len__()-3 if h5 else training_path.__len__()-4])


def set_name(val):
    global name
    name = val
    name_label.config(text=name)


def set_neuronsnum(val):
    global neuronsnum
    neuronsnum = val
    neurons_label.config(text="{} neurons".format(neuronsnum))
    reset_button.config(state='disabled' if neuronsnum == 0 else 'normal')
    if neuronsnum > 0 and training_path != "" and results_path != "":
        train_button.config(state='normal')
    else:
        train_button.config(state='disabled')


def set_current_benchmark(val):
    current_benchmark = val


def reset_benchmarks():
    global neurons
    global neuronsnum
    neurons = [[]]
    set_neuronsnum(0)
    set_current_benchmark(0)


def convert():
    global training_path
    global results_path
    hpelm.make_hdf5(training_path, training_path[:-4] + ".h5", delimiter=',')
    hpelm.make_hdf5(results_path, results_path[:-4] + ".h5", delimiter=',')
    training_path = training_path[:-4] + ".h5"
    results_path = results_path[:-4] + ".h5"
    change_h5(True)


def change_h5(val):
    global h5
    h5 = val
    if val is True:
        convert_button.config(state='disabled')

        train_button.config(text="Train (big data)")
        if neuronsnum > 0:
            train_button.config(state='normal')
        else:
            train_button.config(state='disabled')
    elif val is False:
        convert_button.config(state='normal')

        train_button.config(text="Train (small data)")
        if neuronsnum > 0:
            train_button.config(state='normal')
        else:
            train_button.config(state='disabled')
    elif val is None:
        convert_button.config(state='disabled')
        train_button.config(state='disabled')
        train_button.config(text="Train")


def add_neurons():
    n = int(number_spin.get())
    if n > 0:
        neurons[current_benchmark].append((n, functions[function_names.index(function.get())]))
        set_neuronsnum(neuronsnum + n)


def start():
    pass


name_label = Label(root, text=name)
name_label.pack()

file_select_button = Button(root, text="Select data", command=file_select)
file_select_button.pack()

convert_button = Button(root, text="Convert to hdf5", state='disabled', command=convert)
convert_button.pack()

percentage_label = Label(root, text="Training percentage:")
percentage_label.pack()

percentage_spin = Spinbox(root, from_=10, to=90, increment=5)
percentage_spin.delete(0, "end")
percentage_spin.insert(0, 80)
percentage_spin.pack()

Frame(root, height=20).pack()

neurons_label = Label(root, text="0 neurons")
neurons_label.pack()

number_label = Label(root, text="Number of neurons:")
number_label.pack()
number_spin = Spinbox(root, from_=0, increment=50, to=100000)
number_spin.delete(0, "end")
number_spin.insert(0, 50)
number_spin.pack()

type_label = Label(root, text="Activation function:")
type_label.pack()
type_menu = OptionMenu(root, function, *function_names)
type_menu.pack()

add_button = Button(root, text="Add neurons", state='normal', command=add_neurons)
add_button.pack()

reset_button = Button(root, text="Reset all benchmarks", state='disabled', command=reset_benchmarks)
reset_button.pack()

Frame(root, height=20).pack()

train_button = Button(root, text="Run benchmarks", state='disabled', command=start)
train_button.pack()


root.mainloop()

