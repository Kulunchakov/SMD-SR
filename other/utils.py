from numpy import ceil, dot, int32, floor, array, savetxt, loadtxt, \
                    where, min, interp, arange, sum, sqrt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import draw.main as draw
import numpy as np

import datetime, time, os, sys, string, time
import re, random, math, imp, pickle
import pywt, pywt.data

from scipy.io.wavfile import read
from numpy.random import choice
from functools    import partial
from ast          import literal_eval
from mat4py       import loadmat
from math         import ceil
from utils        import *
from pydub        import AudioSegment
from pydub.utils  import get_array_type

norm = np.linalg.norm

def convex(w, a, b):
    return w*a + (1.0-w)*b
    
def is_int(s):
    val = literal_eval(s)
    return isinstance(val, int) or (isinstance(val, float) and val.is_integer())

def lasso_data_gen(ind_exp, oracle, budget):
    np.random.seed(ind_exp)
    X, y = np.zeros((budget, oracle.p['dim'])), np.zeros(budget)
    p = oracle.objective.p
    for i in range(budget):
        phi = oracle.generate_regressor();
        oracle.calls = oracle.calls + 1
        X[i,:] = phi
        y[i] = phi.dot(oracle.objective.x_optimum) + p['sigma'] * np.random.randn(1)[0]
    return X, y

def get_aggregated_results(p, mode, beta, d):
    name = str(p['mu']) + "_" + str(p['L']) + "_" + str(d) + "_" + str(p['s']) + "_"
    if beta:
        name += str(p['s']) + "_" + str(p['sigma']) + "_" + str(p['noise']) + "_" + str(abs(beta))[:5] + "_{}_".format(mode)
    else:
        name += str(p['s']) + "_" + str(p['sigma']) + "_" + str(p['noise']) + "_{}_".format(mode)
    name = "agg_" + name
    mean = [f for f in get_files("results") if starts(efn(f), name) and "_m.txt" in f]
    assert len(mean) == 1; mean = np.loadtxt(mean[0])
    std = [f for f in get_files("results") if starts(efn(f), name) and "_std.txt" in f]
    assert len(std) == 1; std = np.loadtxt(std[0])
    return mean, std

def update_mean_std(file_prefix, evolution):
        root = "results/"; lenght = len(evolution)

        mean_file = [f for f in get_files(root) if starts(efn(f), file_prefix) and \
                                                            "_m.txt" in f]
        if len(mean_file) == 1:
            num_exp = int(efn(mean_file[0]).split("_")[-2])
            mean = loadtxt(mean_file[0])
        elif len(mean_file) == 0:            
            mean_file = None; num_exp = 0
            mean = np.zeros(lenght)
        else:
            raise ValueError

        weight_1 = num_exp / (num_exp + 1.0)
        new_mean = weight_1 * mean + (1.0 - weight_1) * evolution


        std_file = [f for f in get_files(root) if starts(efn(f), file_prefix) and \
                                                            "_std.txt" in f]
        if len(std_file) == 1:
            std = loadtxt(std_file[0])
        elif len(std_file) == 0:            
            std_file = None
            std = np.zeros(lenght)
        else:
            raise ValueError

        new_std = weight_1 * (std ** 2 + mean**2) + (1.0 - weight_1) * (evolution**2)
        new_std -= new_mean ** 2
        new_std = np.sqrt(new_std)
        
        new_base_name = root + file_prefix + str(num_exp + 1)
        savetxt(new_base_name + "_m.txt", new_mean)
        savetxt(new_base_name + "_std.txt", new_std)
        if mean_file is not None: os.remove(mean_file[0])
        if std_file is not None: os.remove(std_file[0])

def compare(pack, keys, colors=['r','b','g'], legends=[], loglog=1, name=''):
    assert len(colors) == len(legends)
    if len(keys):
        evolutions = [extract_history(h, k) for h,k in zip(pack, keys)]
    else:
        evolutions = pack[:]
    draw.plot_compare_pack(evolutions, colors=colors, legend_titles=legends, 
                            loglog=loglog, name=name)

def compare2(hist_1, hist_2, key_1, key_2, legends):
    evol_1  = extract_history(hist_1, key=key_1)
    evol_2  = extract_history(hist_2, key=key_2)
    draw.plot_compare(evol_1, evol_2, legend_titles=legends)

def represent(history, key):
    evolution  = extract_history(history, key='f(xk)')
    draw.plot_result(evolution)

def convex(w, a, b):
    return w*a + (1.0-w)*b

def sparsify(vector, s):
    data = my_range(len(vector))
    data = sorted(data, key=lambda x: -abs(vector[x]))
    res = np.zeros_like(vector)
    for i in data[:s]:
        res[i] = vector[i]
    return res

def aggregate_logs(logs_dict):
    optima, output, seed_counter, var_auxiliary = dict(), dict(), defaultdict(int), defaultdict(list)
    for key in logs_dict:
        safe_add(output, k=erase_seed(key), v=logs_dict[key])
        seed_counter[erase_seed(key)] += 1
        var_auxiliary[erase_seed(key)].append(logs_dict[key])
    
    for key in var_auxiliary:
        var_auxiliary[key] = array(var_auxiliary[key])
        optima[key] = np.min(var_auxiliary[key])
        var_auxiliary[key] = sum((var_auxiliary[key] - (output[key] / seed_counter[key]))**2, axis=0)
        # print(var_auxiliary[key][:,0])
        var_auxiliary[key] /= seed_counter[key]
        var_auxiliary[key]  = sqrt(var_auxiliary[key])
        
    for key in output:
        output[key] /= seed_counter[key]
    return optima, output, var_auxiliary


def plain(vec):
    return array([i[0] for i in vec])
    
def upd_keys(d):
    a = dict()
    for k in d:
        a[k.replace("test\\", "")] = d[k]
    return a

def interpolate(dict_log, nvals, trimming):
    ticks, vals = extraction(dict_log, 'ticks'), extraction(dict_log, 'logs_exp')
    
    true_ticks  = arange(0,trimming*nvals,step=nvals)
    true_values = array([interp(a, ticks, vals) for a in true_ticks])
    return true_values

def get_logs(fnm, trimming):
    values = extraction(loadmat(fnm), 'logs_exp') 
    return values

def prepare_vars_2_plot(lines, explicit=None):
    length = min(list(map(lambda x: len(x), lines)))
    if explicit is not None:
        length=min([length, explicit])
    lines = list(map(lambda x: x[:length], lines))

    return lines

def prepare_lines_2_plot(lines, optimum=None):
    length = min(list(map(lambda x: len(x), lines)))
    lines = list(map(lambda x: x[:length], lines))
    if optimum is None:
        optimum = min(lines) * 0.996
        
    results = list(map(lambda x: x  - optimum, lines))
    return results

def erase_seed(name):
    return re.sub(r'seed\d+_', '', name).replace("exps\\", "")

# def aggregate_logs(logs_dict):
#     output, seed_counter = dict(), defaultdict(int)
#     for key in logs_dict:
#         safe_add(output, k=erase_seed(key), v=logs_dict[key])
#         seed_counter[erase_seed(key)] += 1
#     for key in output:
#         output[key] /= seed_counter[key]
#     return output 

def is_file_2_layout(file, layout):
    meta = name_2_meta(file)
    for k in layout:
        if type(layout[k])==int and str(meta[k])!=str(layout[k]): return False
        elif type(layout[k])==list and int(meta[k]) not in layout[k]: return False
        else: pass
    return True

def extraction(data,key="logs_exp"):
    data = array([a[0] for a in data[key]])
    return data

def safe_add(d, k, v):
    if k in d: d[k] += v
    else: d[k] = v.copy()

def name_2_meta(name):
    keys = ["data", "setting", "dropout", "lambda", "n", "init_epochs", "seed", "loss"]
    return {k:o for (o,k) in zip(re.findall('\d+', name), keys)}

def name_2_method(name):
    return dict_methods[int(name_2_meta(name)["setting"])]

def meta_2_name(meta, exclude_seed=0):
    keys = ["data", "setting", "dropout", "lambda", "n", "init_epochs", "seed", "loss"]
    if exclude_seed:
        numbers = [meta[k] for k in keys if k != "seed"]
        name = "exp_data{}_s{}_d{}_l{}_n{}_{}_loss{}".format(*numbers)
    else:
        numbers = [meta[k] for k in keys]
        name = "exp_data{}_s{}_d{}_l{}_n{}_{}_seed{}_loss{}".format(*numbers)
    return name


def timer_decorator(func):
    def wrapper(*args, **k_wargs):
        start = time.time()
        returned = func(*args, **k_wargs)
        elapsed = time.time() - start
        if 'trace' in args[0] and args[0].trace == 1:
            args[0].history[-1][func.__name__] = elapsed
        return returned
    return wrapper


def extract_history(history, key='xk'):
    patch = history[1]
    values = []
    for a in history:
        values.extend([a[key] for _ in range(a['copy'])])
    return array(values)

def check_array_one_dimensional(array):
    return len(array.shape) == 1 or max(array.shape) == 1


def common_shorten(sequences, optimum=None, mask=None):
    common_len = min([len(l) for l in sequences])
    if optimum is not None:
        return tuple(convert(l[:common_len], optimum) if m else l[:common_len] for (l, m) in zip(sequences, mask))
    else:
        return tuple(l[:common_len] for l in sequences)


def convert(evolution, optimum_value):
    return (evolution / optimum_value) - 1


def my_dot(a, b):
    a, b = a.reshape((1,-1)), b.reshape((-1,1))
    return dot(a, b)


def bunch_to_generator(bunch):
    n_samples, n_features = bunch.data.shape
    for i in range(n_samples):
        yield bunch.data[i, :].toarray()[0]


def count_lines(filename):
    count = 0
    for _ in open(filename):
        count += 1
    return count

def sq(x):
    return x * x

def int_ceil(arg):
    return int(ceil(arg))


def int_floor(arg):
    return int(floor(arg))


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def store_aggregated_results(p, budget, mode, beta, do_median=False):
    # store_aggregated_results(0, abs(beta))
    # store_aggregated_results(1, abs(0.05))
    # store_aggregated_results(2, abs(0.05))
    # store_aggregated_results(1, str(100.0))
    # store_aggregated_results(2, str(100.0))
    name  = str(p['mu']) + "_" + str(p['L']) + "_" + str(p['dim']) + "_" + str(p['s']) + "_"
    name += str(p['s']) + "_" + str(p['sigma']) + "_" + str(p['noise']) + "_"
    if beta:
        name += str(beta)[:5] + "_{}_".format(mode)
    else:
        name += "{}_".format(mode)
    files = [f for f in os.listdir("exps") if starts(efn(f), name)]
    if len(files)==0: 
        print("name:", name); print("no files!"); return
        
    files_0 = files[:]
    budget = np.median([len(np.loadtxt("exps/" + f)) for f in files_0])
    budget = int(budget)
    files = []
    for f in files_0:
        if len(np.loadtxt("exps/" + f)) >= budget:
            files.append(f)
    print(len(files), "files collected.")
    # budget = np.inf
    # for (i,f) in enum(files): 
    #     data_loaded = np.loadtxt("exps/" + f)
    #     budget = min([len(data_loaded), budget])

    # budget = np.loadtxt("exps/" + files[0]).size
    arr = np.zeros((len(files), budget))
    to_delete = []
    for (i,f) in enum(files): 
        arr[i,:] = np.loadtxt("exps/" + f)[:budget]
        to_delete.append("exps/" + f)
    name = "agg_" + name + str(len(files))
    if not do_median:
        m, std = arr.mean(axis=0), array(arr).std(axis=0)
        np.savetxt("av_exps/" + name + "_m.txt", m)
        np.savetxt("av_exps/" + name + "_std.txt", std)
    else:
        m, q_025, q_075 = np.median(arr, axis=0), \
                          np.quantile(arr, 0.25, axis=0), np.quantile(arr, 0.75, axis=0)
        np.savetxt("av_exps/" + name + "_med.txt", m)
        np.savetxt("av_exps/" + name + "_q_025.txt", q_025)
        np.savetxt("av_exps/" + name + "_q_075.txt", q_075)
    # for f in to_delete:
    #     os.remove(f)
    # print("deleted", len(to_delete), "files")

# notches = np.hstack((np.arange(10, 500, 20), 
#                      np.arange(600, 2500, 50), 
#                      np.arange(2600, 10000, 200)))

notches = np.hstack((np.arange(10, 2500, 300), 
                     np.arange(3600, 10000, 2000)))

notches_small = np.hstack((np.arange(10, 500, 100), 
                         np.arange(600, 2500, 500), 
                         np.arange(2600, 10000, 1000)))


class Models:
    def __init__(self):
        import models.mirror_descent
        import models.catalyst
        import models.catalyst_base
        import models.loc_mirror_descent
        import models.acc_mirror_descent

        self.MirrorDescent = models.mirror_descent.MirrorDescent
        self.CatalystBase = models.catalyst_base.CatalystBase
        self.Catalyst = models.catalyst.Catalyst
        self.LOC_MirrorDescent = models.loc_mirror_descent.LOC_MirrorDescent
        self.AC_MirrorDescent = models.acc_mirror_descent.ACC_MirrorDescent

def plot_restart_vs_lasso_basic(p, data_restart, data_lasso):
    data_restart_std = np.zeros_like(data_restart)
    data_lass_std    = np.zeros_like(data_lasso)
    pack, pack_var = [data_restart, data_lasso], [data_restart_std, data_lass_std]
    pack = common_shorten(pack); pack_var = common_shorten(pack_var)
    codes = [2, 3, 1]
    colors = [line_colors[c] for c in codes]; colors_var = [face_colors[c] for c in codes]
    legend_titles = ['(rest) hat(xk) evolution', 
                     'lasso evolution']
    name = str(p['mu']) + "_" + str(p['L']) + "_" + str(p['dim']) + "_" + str(p['s']) + "_"
    name += str(p['sigma']) + "_" + str(p['seed']) + "_1"
    final_plots(pack, colors, pack_var, colors_var, legend_titles, name)

def plot_restart_vs_lasso_basic_3(p, data_acc, data_md, data_base):
    data_acc_std = np.zeros_like(data_acc)
    data_md_std    = np.zeros_like(data_md)
    data_base_std    = np.zeros_like(data_base)
    pack = [data_acc, data_md, data_base]
    pack_var = [data_acc_std, data_md_std, data_base_std]

    pack = common_shorten(pack); pack_var = common_shorten(pack_var)
    codes = [2, 3, 1]
    colors = [line_colors[c] for c in codes]; colors_var = [face_colors[c] for c in codes]
    legend_titles = ['accelerated restart', 
                     'base restart',
                     'acc. mirror descent']
    name = str(p['mu']) + "_" + str(p['L']) + "_" + str(p['dim']) + "_" + str(p['s']) + "_"
    name += str(p['sigma']) + "_" + str(p['seed']) + "_1"
    final_plots(pack, colors, pack_var, colors_var, legend_titles, name)


def plot_restart_vs_md(p, mu, sigma, beta, beta_2=None):
    p['mu'], p['sigma'] = mu, sigma
    data_m_0, data_std_0 = get_aggregated_results(p, 0, abs(beta), 1000)
    data_m_1, data_std_1 = get_aggregated_results(p, 1, abs(beta if beta_2 is None else beta_2), 1000)
    data_m_2, data_std_2 = get_aggregated_results(p, 2, abs(beta if beta_2 is None else beta_2), 1000)
    
    pack, pack_var = [data_m_0, data_m_1, data_m_2], [data_std_0, data_std_1, data_std_2]
    pack = common_shorten(pack); pack_var = common_shorten(pack_var)
    
    codes = [2, 3, 1]
    colors = [line_colors[c] for c in codes]; colors_var = [face_colors[c] for c in codes]
    legend_titles = ['(rest) hat(xk) evolution', 
                     '(md) hat_xk evolution', '(md) xk evolution', 'lasso']
    final_plots(pack, colors, pack_var, colors_var, legend_titles)

def plot_restart_vs_lasso(p, mu, sigma, beta):
    p['mu'], p['sigma'] = mu, sigma
    data_m_0, data_std_0 = get_aggregated_results(p, 0, abs(beta), 6000)
    data_m_1, data_std_1 = get_aggregated_results(p, 7, "", 6000)
    pack, pack_var = [data_m_0, data_m_1], [data_std_0, data_std_1]
    pack = common_shorten(pack); pack_var = common_shorten(pack_var)
    codes = [2, 3, 1]
    colors = [line_colors[c] for c in codes]; colors_var = [face_colors[c] for c in codes]
    legend_titles = ['(rest) hat(xk) evolution', 
                     'lasso evolution']
    final_plots(pack, colors, pack_var, colors_var, legend_titles)

def final_plots(pack, colors, pack_var, colors_var, legend_titles, name):
    import matplotlib.patches as mpatches
    import matplotlib

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.figure(figsize=(12, 6)); plt.title("Comparison of algorithms")

    for line, c, var, var_c in zip(pack, colors, pack_var, colors_var):
        plt.plot(line, lw=1.3, c=c)
        plt.fill_between(np.arange(len(line)), line-var, line+var, alpha=0.2, edgecolor='#CC4F1B', 
                           facecolor=var_c, lw=0)
        plt.yscale( "log" )

    plt.xlabel("# iteration")
    plt.ylabel(r"$\log\left( f/f* - 1\right)$ ")
    plt.rcParams['grid.alpha'] = 0.4
    plt.grid(1)
    plt.legend(handles=[mpatches.Patch(color=c, label=lt) for c, lt in zip(colors, legend_titles)])
    plt.rcParams.update({'font.size': 14})
    plt.savefig("figures/" + name+'.png')
    plt.show()

def find_size_of_vec_from_pyr(coeffs):
    output, sizes = 0, []
    for l in coeffs:
        if type(l) == tuple:
            output += np.sum(j.size for j in l)
            sizes.append((len(l), [j.shape for j in l]))
        else:
            output += l.size
            sizes.append((l.shape,))
    return output, sizes

def convert_pyr_2_vec(coeffs):
    vec_len, sizes = find_size_of_vec_from_pyr(coeffs)
    output, cur_pos = np.zeros(vec_len), 0
    for l in coeffs:
        if type(l) == tuple:
            for j in l:
                output[cur_pos:cur_pos+j.size] = j.flatten()
                cur_pos += j.size
        else:
            output[cur_pos:cur_pos+l.size] = l.flatten()
            cur_pos += l.size
    return output, sizes

def convert_vec_2_pyr(vec, sizes):
    output, cur_pos = [], 0
    for ent in sizes:
        if len(ent) == 1:
            shape, n_elems = ent[0], np.prod(ent[0])
            output.append(np.array(vec[cur_pos:cur_pos+n_elems]).reshape(shape))
            cur_pos += n_elems
        else:
            output.append([])
            for sub_ent in ent[1]:
                shape, n_elems = sub_ent, np.prod(sub_ent)
                output[-1].append(np.array(vec[cur_pos:cur_pos+n_elems]).reshape(shape))
                cur_pos += n_elems
            output[-1] = tuple(output[-1])

    return output
    
def print_error(label, value):
    print(label.ljust(12) + "{:.6E}".format(value))
# def pyr_2_vec_haar(pyr):
#     temp = np.cumsum([np.sum(j.size for j in i) for i in pyr])
#     output = np.zeros(temp[-1])
#     counter = 0
#     for level in pyr:
#         if counter==0:
#             output[counter] = level[0][0]; counter += 1; continue
#         for ent in level:
#             for row in range(ent.shape[0]):
#                 for col in range(ent.shape[1]):
#                     output[counter] = ent[row][col]; counter += 1
#     return output


# def vec_2_pyr_haar(vec):
#     # n_levels = int(np.log(vec.size) / np.log(2))
#     output = []
#     n_level = 0; cur_pos = 0
#     while cur_pos < vec.size:
#         if cur_pos == 0:
#             output.append(np.array([[vec[0]]])); cur_pos+=1; continue
#         to_append = []
#         for _ in range(3):
#             cur_ent = np.array(vec[cur_pos:cur_pos + 4**n_level]).reshape((2**n_level, 2**n_level))
#             cur_pos += 4**n_level
#             to_append.append(cur_ent)
#         output.append(tuple(to_append))
#         n_level += 1
#     return output


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


line_colors = { 1: 'm', 2: 'blue', 3: 'g', 5: 'red', 6: 'darkblue', 7: 'brown', 8: 'blue', 
            9: "lime", 11: "red", 12: 'darkblue', 17: "darkviolet", 18: "lime", 19: "darkviolet"}
edge_colors = { 1: 'indianred', 2: 'blue', 3: 'lime', 5: 'red', 6: 'darkblue', 7: 'indianred', 8: 'blue', 
            9: "lime", 11: "red", 12: 'darkblue', 17: "darkviolet", 18: "lime", 19: "darkviolet"}
face_colors = { 1: 'mistyrose', 2: 'lightblue', 3: 'beige', 5: 'peachpuff', 6: 'darkblue', 7: 'mistyrose', 
               8: 'lightsteelblue', 9: "beige", 11: "peachpuff", 12: 'lightsteelblue', 17: "lightpink"}
face_colors = { 1: 'lightcoral', 2: 'lightblue', 3: 'beige', 5: 'peachpuff', 6: 'darkblue', 7: 'lightcoral', 
               8: 'lightsteelblue', 9: "palegreen", 11: "peachpuff", 12: 'lightsteelblue', 17: "lightpink", 
               18: "palegreen", 19: "peachpuff"}
