# coding=utf-8
import benchmark as bn
import preprocessor as pp
import hpelm
import matplotlib.pyplot as plt
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
neurons = [[]]
current_benchmark = 0
neuronsnum = 0
neuronsnums = [0]
functions = ['lin', 'sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf']
function_names = ['Linear', 'Sigmoid', 'Hyperbolic tangent', 'RBF (L1)', 'RBF (L2)', 'RBF (Linf)']
groups = [0]
last_group = 0

root = Tk()
root.title("ELM")
root.geometry("220x510")
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
        set_name(training_path[ind + 1:training_path.__len__() - 3 if h5 else training_path.__len__() - 4])


def set_name(val):
    global name
    name = val
    name_label.config(text=name)


def add_neuronsnum(val):
    global neuronsnum
    global neuronsnums
    neuronsnum += val
    neuronsnums[current_benchmark] += val
    neurons_label.config(text="{} neurons".format(neuronsnums[current_benchmark]))
    reset_button.config(state='normal' if neuronsnum > 0 or neurons.__len__() > 1 else 'disabled')
    if training_path != "" and results_path != "":
        correct = True
        for n in neuronsnums:
            if n == 0:
                correct = False
                break
        train_button.config(state='normal' if correct else 'disabled')
    else:
        train_button.config(state='disabled')


def set_current_benchmark(val):
    global current_benchmark
    global last_group
    current_benchmark = val
    if current_benchmark >= neurons.__len__():
        current_benchmark = neurons.__len__() - 1
    if current_benchmark < 0:
        current_benchmark = 0
    add_neuronsnum(0)
    last_group = groups[current_benchmark]
    benchmark_label.config(text='Benchmark {}/{}'.format(current_benchmark + 1, neurons.__len__()))
    benchmark_previous.config(state='disabled' if current_benchmark == 0 else 'normal')
    benchmark_next.config(state='disabled' if current_benchmark + 1 == neurons.__len__() else 'normal')
    group_spin.delete(0, "end")
    group_spin.insert(0, groups[current_benchmark] + 1)


def next_benchmark():
    set_current_benchmark(current_benchmark + 1)


def previous_benchmark():
    set_current_benchmark(current_benchmark - 1)


def reset_current_benchmark():
    global neurons
    global neuronsnum
    global neuronsnums
    neurons[current_benchmark] = []
    add_neuronsnum(-neuronsnums[current_benchmark])


def reset_benchmarks():
    global neurons
    global neuronsnum
    global neuronsnums
    global groups
    global last_group
    neurons = [[]]
    neuronsnum = 0
    neuronsnums = [0]
    groups = [0]
    set_current_benchmark(0)
    add_neuronsnum(0)


def add_benchmark():
    global neurons
    global neuronsnum
    global neuronsnums
    global current_benchmark
    neurons.insert(current_benchmark + 1, [])
    neuronsnums.insert(current_benchmark + 1, 0)
    groups.insert(current_benchmark + 1, last_group)
    set_current_benchmark(current_benchmark + 1)


def delete_benchmark():
    global neurons
    global neuronsnum
    global neuronsnums
    global current_benchmark
    if neurons.__len__() > 1:
        neurons.__delitem__(current_benchmark)
        neuronsnum -= neuronsnums[current_benchmark]
        neuronsnums.__delitem__(current_benchmark)
        groups.__delitem__(current_benchmark)
        set_current_benchmark(current_benchmark)
    else:
        reset_benchmarks()


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

        train_button.config(text="Run benchmarks (big data)")
        if neuronsnum > 0:
            train_button.config(state='normal')
        else:
            train_button.config(state='disabled')
    elif val is False:
        convert_button.config(state='normal')

        train_button.config(text="Run benchmarks (small data)")
        if neuronsnum > 0:
            train_button.config(state='normal')
        else:
            train_button.config(state='disabled')
    elif val is None:
        convert_button.config(state='disabled')
        train_button.config(state='disabled')
        train_button.config(text="Run benchmarks")


def add_neurons():
    n = int(number_spin.get())
    if n > 0:
        neurons[current_benchmark].append((n, functions[function_names.index(function.get())]))
        add_neuronsnum(n)


def set_group():
    global groups
    global last_group
    groups[current_benchmark] = last_group = int(group_spin.get()) - 1


def start():
    percentage = int(percentage_spin.get())
    if percentage < 10:
        percentage = 10
    if percentage > 90:
        percentage = 90
    percentage /= 100.0
    if h5 is True:
        benchmark = bn.Benchmark.big_benchmark(name, percentage, neurons,
                                               training_path, results_path,
                                               training_path[:-3] + "_test_results" + ".h5")
    else:
        benchmark = bn.Benchmark.small_benchmark(name, percentage, neurons,
                                                 pp.open_csv(training_path), pp.open_csv(results_path))
    errors, percentages, times = benchmark.run()
    # win = Toplevel()
    # win.title = "Results:"
    # x_label = Label(win, text="Neurons: {}".format(neuronsnums))
    # x_label.pack()
    # err_label = Label(win, text="Mean square error: {}".format(errors))
    # err_label.pack()
    # per_label = Label(win, text="Correct ratio: {}".format(percentages))
    # per_label.pack()
    # times_label = Label(win, text="Training time: {}".format(times))
    # times_label.pack()

    grps = set(groups)
    fig, (axe, axp, axt) = plt.subplots(nrows=3)

    for g in grps:
        xd = [x for ind, x in enumerate(neuronsnums) if groups[ind] == g]
        e = [x for ind, x in enumerate(errors) if groups[ind] == g]
        p = [x for ind, x in enumerate(percentages) if groups[ind] == g]
        t = [x for ind, x in enumerate(times) if groups[ind] == g]

        axe.plot(xd, e, label='Group {}'.format(g))
        axp.plot(xd, p, label='Group {}'.format(g))
        axt.plot(xd, t, label='Group {}'.format(g))

    axe.set_title('Mean square error')
    axe.set_xlabel('Number of neurons')
    axe.set_ylabel('Mean square error')

    axp.set_title('Correct ratio')
    axp.set_xlabel('Number of neurons')
    axp.set_ylabel('Correct ratio')

    axt.set_title('Training time')
    axt.set_xlabel('Number of neurons')
    axt.set_ylabel('Training time')

    plt.legend()
    plt.subplots_adjust(hspace=1)
    plt.show()


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

benchmark_frame = Frame(root)
benchmark_frame.pack()
benchmark_previous = Button(benchmark_frame, text="<", command=previous_benchmark, state='disabled')
benchmark_previous.grid(row=0, column=0)
benchmark_label = Label(benchmark_frame, text="Benchmark 1/1")
benchmark_label.grid(row=0, column=1)
benchmark_next = Button(benchmark_frame, text=">", command=next_benchmark, state='disabled')
benchmark_next.grid(row=0, column=2)

neurons_label = Label(root, text="0 neurons")
neurons_label.pack()

group_label = Label(root, text="Group:")
group_label.pack()

group_spin = Spinbox(root, from_=1, to=100000, command=set_group)
group_spin.delete(0, "end")
group_spin.insert(0, 1)
group_spin.pack()

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

Frame(root, height=20).pack()

benchmark_add = Button(root, text="Add benchmark", command=add_benchmark)
benchmark_add.pack()

benchmark_delete = Button(root, text="Delete benchmark", command=delete_benchmark)
benchmark_delete.pack()

benchmark_reset = Button(root, text="Reset benchmark", command=reset_current_benchmark)
benchmark_reset.pack()

Frame(root, height=20).pack()

reset_button = Button(root, text="Reset all benchmarks", state='disabled', command=reset_benchmarks)
reset_button.pack()

train_button = Button(root, text="Run benchmarks", state='disabled', command=start)
train_button.pack()

root.mainloop()
