from subprocess import DEVNULL, STDOUT, check_call
from collections import defaultdict, Counter
import datetime, time, os, string, imp, subprocess
import requests, glob, re, urllib, shutil, zipfile
import collections, pickle, sys, random, itertools
from functools import reduce, partial
import webbrowser, json

import matplotlib.pyplot as plt
import pyperclip as ppc

from pathlib import Path
from pprint  import pprint
from numpy   import random
from datetime import timedelta  
from numpy.random import choice
from IPython.display import HTML
import numpy as np

dt = np.dot
log = np.log
dto = np.outer
rank = np.linalg.matrix_rank
mnorm = np.random.rand
normal = np.random.normal
normal = np.random.normal
binom = partial(np.random.binomial, n=1, p=0.5)
norm = np.linalg.norm
nin = partial(np.linalg.norm, ord=2)
tr = np.trace
arr = np.array

myid = 44774089
os_sep = "/"

#######################################
################# OS ##################
#######################################

def pjoin(*args):
    return os.path.join(*args)

def go_up_dir(directory):
    return str(Path(directory).parents[0]) + os_sep

def listdir(directory):
    return os.listdir(directory)

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_dir_chain(directory):
    auxiliary = directory
    subfolders = []
    while auxiliary != ".":
        subfolders.append(auxiliary)
        auxiliary = go_up_dir(auxiliary)
    subfolders = subfolders[::-1]
    for sf in subfolders:
        check_dir(sf)
        
def create_file(path):
    head, tail = os.path.split(path)
    if head:
        check_dir_chain(head)
    if not os.path.exists(path):
        open(path, 'w+').close()

###################################
########## DICTIONARIES ###########
###################################

def enum(*args, **kwargs):
    return enumerate(*args, **kwargs)
def create_file(filename):
    if not os.path.exists(filename):
        open(filename, 'w+',)
def inf(filename, encoding="utf-8", to_strip=True, ignore=0):
    return input_file(filename, encoding=encoding, to_strip=to_strip, ignore=ignore)
def input_file(filename, encoding="utf-8", to_strip=True, ignore=0):
    errors = 'ignore' if ignore else None
    if to_strip:
        return [l.strip() for l in open(filename, encoding=encoding, errors=errors)]
    else:
        return [l for l in open(filename, encoding=encoding, errors=errors)]

def input_solid_text(filename, encoding="utf-8", ignore=0):
    data = input_file(filename, encoding=encoding, ignore=ignore)
    return '\n'.join(data)
def check_absence(filename, obj):
    return obj not in input_file(filename)
def check_presence(filename, obj):
    if not os.path.exists(filename):
        open(filename, 'a+').close()
    return obj in input_file(filename)
def safe_key_check(dic, key, value):
    if key in dic and dic[key]==value:
        return True
    return False

def clear_file(filename):
    open(filename, 'w+').close()
def store_string(string, filename, regime='a+'):
    print(string, file=open(filename, regime, encoding='utf-8'))
def store_strings(strings, filename, regime='a+'):
    print("\n".join(strings), file=open(filename, regime, encoding='utf-8'))
def store_int(i, filename, regime='w'):
    print(i, file=open(filename, regime), end='\n')
def store_ints(ints, filename, regime='w'):
    store_string('\n'.join(my_map(str, ints)), filename)
def open_utf(filename, regime='w'):
    return open(filename, regime, encoding='utf-8')
def store_pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)
def read_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)



def read_int(filename):
    return int(input_file(filename)[0])
def erase_int(j, filename):
    ints = read_ints(filename)
    file = open(filename, 'w')
    for i in ints:
        if i != j: file.write("{}\n".format(i))
    file.close()
def read_float(filename):
    return float(input_file(filename)[0])
def read_str(filename):
    return input_file(filename)[0]
def read_text(filename):
    return '\n'.join(input_file(filename))
def read_ints(filename):
    return my_map(int, input_file(filename))
def save_object(obj, name):
    with open(name, 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(name):
    with open(name, 'r+b') as f:
        return pickle.load(f)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
def filter_keys(d, keys):
    return {k:d[k] for k in keys}

def filter_pics(files):
    def end(s, ext):
        return s.endswith(ext)
    exts = ['.jpg', '.png', 'jpeg']
    output = [f for f in files if any([end(f, e) for e in exts])]
    return output
def store_instance(filename, instance):
    with open(filename, 'a+', encoding="utf-8") as f:
        f.write(str(instance) + '\n')
def filter_punctuation(s):
    return ''.join(c for c in s if c not in string.punctuation)
def random_key(dic):
    return choice(pkeys(dic, True))
def n_files(root, rec=0):
    n = str(len(get_files(root,rec)))
    print("There are {} files in the category{}.".format(bold(n), " (recursive)" if rec else ""))
def shorten_filenames(root, threshold=8, order=9):
    files = get_files(root)

    for (f, i) in zip(files, my_nrand(len(files), order, cast=str)):
        if len(extract_filename(f)) > threshold:
            rename_tail(f, i)
def rename_tail(f, name):
    if f.split(".")[-1] == name.split(".")[-1]:
        os.rename(f, extract_root(f) + '//' + name)
    else:
        os.rename(f, extract_root(f) + '//' + name + "." + f.split(".")[-1])
def my_rand(order, cast=int):
    return cast(random.randint(10**order, 9*10**order))
def my_nrand(num, order, cast=int):
    import random
    return [cast(i) for i in random.sample(range(1, 10**order), num)]
def my_all(data, condition):
    return all([condition(a) for a in data])
class hashabledict(dict):
        def __lt__(self, item):
            return self['from_id'] < item['from_id']
        def __hash__(self):
            return hash(tuple(sorted(self.items())))

def reverse_dict(d):
    return {item : key for key, item in d.items()}
def safe_check():
    pass

def dict_value(d):
    return list(d.values())[0]
def dict_from2lists(l1, l2):
    return dict(zip(l1,l2))
def dict_slice(d, s=0, e=5):
    keys = pkeys(d, 0)
    return {k: d[k] for k in keys[s:e]}


def print_slice(data, num, delimiter=-1,logging=None, 
                in_logging=None, mid_separator=None):
    if in_logging is not None:
        data = [a for a in data[:num] if not in_logging(a)]
    for a in data[:num]:
        print(a)
        if delimiter>0:
            print_delimiter(delimiter)
    if logging is not None:
        list(map(logging, data[:num]))
    if mid_separator is not None:
        if isinstance(mid_separator, int):
            for i in range(mid_separator):
                print()
        else:
            print(mid_separator)


def print_dict(d, amount = None):
    count = 0
    for (k,v) in d.items():
        print("d[{}]: {}".format(k, v))
        count += 1
        if count == amount: return 

def pickle_dict(d, filename):
    if os.path.exists(filename):
        a = pickle.load(open(filename, "rb+"))
    else:
        a = d
    for k in d:
        a[k] = d[k]

    pickle.dump(a, open(filename, "wb+"))

def store_dict(dic, filename, encoding='utf-8', sep=','):
    with open(filename, 'w', encoding=encoding) as output:
        for key in dic:
            output.write(key + sep + str(dic[key]) + '\n')

def read_dict(filename, key_type = str, value_type = float, separator = ','):
    data = [line.strip() for line in open(filename, encoding="utf-8")]
    dict_values = dict()
    for line in data:
        key, *items = line.split(separator)
        dict_values[key_type(key)] = value_type(separator.join(items))
    return dict_values


def obj_total_size(o, handlers={}, verbose=False):
    from sys import getsizeof, stderr
    from itertools import chain
    from collections import deque
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def read_lines(filename=None, generator=None, n_lines=1, show=True):
    output, count = [], 0
    if filename is not None:
        for line in open(filename):
            if show:
                print(line)
                if count == n_lines: return
            output.append(line)
            count += 1
            if count == n_lines:
                return output if n_lines > 1 else output[0]
    elif generator is not None:
        if type(generator) == Bunch:
            line = generator.data[count].toarray()[0]
        else:
            line = next(generator)
        if show: print(line)
        else: output.append(line)
        count += 1
        if count == n_lines:
            if not show: return output if n_lines > 1 else output[0]
            return None
    else:
        raise ValueError

def count_lines(filename):
    count = 0
    for _ in open(filename):
        count += 1
    return count

def remove_ext(s):
    return '.'.join(s.split(".")[:-1])
def reload(a):
    imp.reload(a)
def rel(mod): 
    imp.reload(mod)
def cur_index(ind):
    print("Process {}".format(bold(ind)))

# def read_dict(filename, type_ = str):
#     data = [line.strip() for line in open(filename)]
#     dict_values = dict()
#     for line in data:
#         key, item = line.split(',')
#         dict_values[key] = type_(item)
#     return dict_values

def bold(text):
    return "{}{}{}".format(mycol.bd, str(text), mycol.e)
def blue(text):
    return "{}{}{}".format(mycol.b, str(text), mycol.e)
def bold_blue(text):
    return bold(blue(text))
def bold_red(text):
    return bold(red(text))
def bold_green(text):
    return bold(green(text))
def red(text):
    return "{}{}{}".format(mycol.r, str(text), mycol.e)
def green(text):
    return "{}{}{}".format(mycol.g, str(text), mycol.e)

def intersect(*args):
    args = list(map(set, args))
    return eval("{}(set.intersection(*args))".format(type(args[0]).__name__))
def unite(*args):
    args = list(map(set, args))
    return eval("{}((set.union(*args)))".format(type(args[0]).__name__))

def first_which(signal, condition):
    pos = 0
    while not condition(signal[pos]):
        pos+=1
        if pos == len(signal): return None
    return pos

def open_ffile(func):
    os.system('subl "{}"'.format(func.__globals__['__file__']))
def open_started():
    for f in input_file("data/start_files.txt"):
        os.system('subl "{}"'.format(f))
def last_which(signal, condition):
    sub_signal = signal[::-1]
    return len(signal) - 1 - first_which(sub_signal, condition)

def myfilter(condition, data):
    return list(filter(condition, data))
def my_filter(condition, data):
    return list(filter(condition, data))
def my_map(function, data):
    return list(map(function, data))
def my_all(function, data):
    return sum(my_map(function, data)) == len(data)
def my_find(function, data):
    result = [i for i in data if function(i)]
    return result[0] if len(result)==1 else result
def my_range(n, st=0, k=1):
    if not isinstance(n,int):
        n = len(n)
    return list(range(st, n, k))
def zero_range(n):
    return [0 for _ in range(n)]
def unique(obj):
    output, seen = [], set()
    for t in obj:
        if t not in seen:
            seen.add(t)
            output.append(t)
    return output
def n_unique(obj):
    return len(set(obj))
def compare_lists(l1, l2):
    return set(l1) == set(l2)

def slog_counter(words):
    l = ['а', 'о', 'э', 'и', 'у', 'ы', 'е', 'ё', 'ю', 'я']
    data = {w: sum([w.lower().count(e) for e in l]) for w in words}
    return data
def dict1_minus_d2(d1, d2):
    return {(k,v) for (k,v) in d1.items() if k in d2 and v == d2[k]}

def get_key_seq(l, key):
    return [a[key] for a in l]

def sort_by_dict_key(data, key, d, decr=1):
    return sorted(data, key=lambda x: (1-2*decr) * d[x[key]])
def sort_by_key(data, key, decr=1):
    return sorted(data, key=lambda x: (1-2*decr) * x[key])
def sort_by_attr(data, decr=1):
    return sorted(data, key=lambda x: (1-2*decr) * getattr(x,key))

def get_list_keys(d, key):
    return [a[key] for a in d]
def get_list_attrs(d, attr):
    return [getattr(a,attr) for a in d]
def d_top(dic, i=0):
    return (lkeys(dic)[i], lvals(dic)[i])
def n_keys(dic):
    return len(lkeys(dic))
def nkeys(dic):
    return len(list(dic.keys()))
def lkeys(dic, amount = None):
    keys = sorted(list(dic.keys()))[:amount]
    return keys
def lvals(dic, amount = None):
    keys = sorted(list(dic.values()))[:amount]
    return keys

def pkeys(dic, show = False, amount = None):
    keys = sorted(list(dic.keys()))[:amount]
    if not show: return keys
    if len(keys)<1000: print(keys)

# def print_error():
#     print_current_time()
#     print("Unexpected error when autorizing:", sys.exc_info()[0])

def time_2_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)
def print_date_full(date = None, time = None):
    if time is not None: date = time_2_datetime(time)
    print(date.strftime('%d.%b %H:%M:%S'))

def print_delimiter(nlines = 2, end = '\n', file = None):
    print(nlines*('-'*55+'\n'), end = end, file = file)    

def safe_remove_files(files, root=''):
    for f in files:
        try: os.remove(root + '/' + f)
        except: os.remove(root + f)
def split_string_by_spaces(s, limit, sep=" "):
    output = []
    for l in s.split("\n"):
        if len(l)==0: 
            output.append("")
            continue
        words = l.split()
        if max(map(len, words)) > limit:
            raise ValueError("limit is too small")
        res, part, others = [], words[0], words[1:]
        for word in others:
            if len(sep)+len(word) > limit-len(part):
                res.append(part)
                part = word
            else:
                part += sep+word
        if part:
            res.append(part)
        output.append("\n".join(res))
    return "\n".join(output)
def b2s(b):
    return b.decode("utf-8",'ignore') 
def bts(b):
    return b.decode("utf-8",'ignore') 
def starts(s, t):
    return s.startswith(t)
def ends(s, t):
    return s.endswith(t)
def find_all_occs(q, s):
    return [m.start() for m in re.finditer(q, s)]
def close_files(files):
    for f in files: f.close()

def find_science_tag(tag, filename, tl=100, tr=30):
    if not isinstance(tag, list): tag = [tag]
    data = open(filename, 'r', errors='ignore').readlines()
    data = " ".join(data).replace("\n", " ")
    res = []
    for t in tag:
        t_0 = t
        t = t.replace("[", "\["); t = t.replace("]", "\]")
        for match in re.finditer(t, data):
            l, r = match.span() #, match.group()
            res.append(data[max(l-tl,0):min(len(data), r+tr)].replace(t_0, bold(t_0)))
    print(len(res))
    for i in unique(res):
        print(i, end="\n\n")

def extracted(num, obj):
    if isinstance(num, collections.Iterable):
        num = len(num)
    if num>1:
        print("Extracted {} {}s.".format(bold_blue(num), obj))
    else:
        print("Extracted {} {}.".format(bold_blue(num), obj))

def change_position_in_split(data, new_value, sep='.', index=-1):
    return sep.join(data.split(sep)[:index]) + sep + new_value + sep + sep.join(data.split(sep)[index+1:])

def get_file_size(filename):
    return os.path.getsize(filename)
def get_directory_size(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += get_directory_size(itempath)
    return total_size

def show_folders_ranked_by_size(root):
    if not ends(root,"/"): root += "/"
    files = os.listdir(root)
    dic = dict()
    for f in files:
        if os.path.isdir(root+f):
            try: dic[f] = get_directory_size(root+f)//1000
            except PermissionError: pass
            except: raise

    tuples = sorted([(-dic[f], f) for f in dic])

    for (_,f) in tuples:
        if os.path.isdir(root+f):
            print("{} : {}".format(f, dic[f]))
def gf(root, recursive=False):
    return get_files(root, recursive)

def gf(root, recursive=False, tails=False):
    return get_files(root, recursive=recursive, tails=tails)
def get_files(root, recursive=False, tails=False):
    if recursive:
        return [os.path.join(dp, f) if not tails else f for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn]
    else:
        return [root + '/' + f if not tails else f for f in os.listdir(root) if os.path.isfile("{}/{}".format(root, f))]
def get_dirs(root, tail=True):
    prefix = root + '/' if tail else ""
    return [prefix + f for f in os.listdir(root) if os.path.isdir("{}/{}".format(root, f))]

def ef(*args, **kwargs):
    return extract_filename(*args, **kwargs)
def extract_filename(path, ext=False):
    if "." in path.split('/')[-1]:
        output = '.'.join(path.split('/')[-1].split('.')[:-1] )
    else:
        output = path.split('/')[-1]
    return output + "." + path.split('.')[-1] if ext else output
def extract_root(path):
    return '/'.join(path.split('/')[:-1])

def change_ext(file, new_ext):
    if '.' not in new_ext:
        new_ext = '.' + new_ext
    return extract_root(file) + '/' + extract_filename(file) + new_ext

def get_size(file):
    return os.stat(file).st_size
def is_empty(file):
    return get_size(file)==0
def do_empty(filename):
    open(filename, 'w').close()
def is_nonempty(file):
    return get_size(file)!=0

def filling(arg, func):
    pool = []
    for n in arg:
        pack = func(n)
        pool.extend(pack)
    return pool
    
def change_ext_and_save(f, new_ext):
    os.rename(f,  change_ext(f, new_ext))

def mass_change_ext(root, old_ext, new_ext):
    files = get_files(root)
    for f in files:
        if ends(f, old_ext):
            change_ext_and_save(f, new_ext)


class mycol:
    h = '\033[95m'
    b = '\033[94m'
    g = '\033[92m'
    w = '\033[93m'
    r = '\033[91m'
    e = '\033[0m'
    bd = '\033[1m'
    u = '\033[4m'
    y = '\033[0;33m'

def noerror_decorator(function_to_decorate):
    def wrapper(*args, **kwargs):
        try: output = function_to_decorate(*args, **kwargs)
        except: print("Unexpected error when autorizing:", sys.exc_info()[0]); return None
        return output
    return wrapper

def noerror_decorator_silent(function_to_decorate):
    def wrapper(*args, **kwargs):
        try: output = function_to_decorate(*args, **kwargs)
        except: return None
        return output
    return wrapper

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
        label.value = ""
        progress.value = index


def binary_search(a, x, lo=0, hi=None):
    from bisect import bisect_left
    hi = hi if hi is not None else len(a)
    pos = bisect_left(a, x, lo, hi) 
    return (pos if pos != hi and a[pos] >= x else -1)



def uclose(files):
    for f in files:
        f.close()



def mwait():
    wait(0.5, scale = 0)
def ec_wait(st):
    wait(max(0.4 - (cur_time() - st), 0), scale = 0)
def wait(amount, scale = 0):
    time.sleep(abs(amount + random.rand(1)[0]*scale))

def filter_data_through_data(a, b):
    res, seen = [], set(list(b))
    for i in a:
        if i not in seen:
            seen.add(i)
            res.append(i)
def get_current_time():
    return time.strftime("%H:%M:%S", time.localtime())
def print_current_time():
    print(get_current_time(), end = '; ')

def mass_convert2jpg(root=r"C:\Users\user\Downloads\publishing"):
    files = my_filter(is_picture,os.listdir(root))
    for f in files:
        if ends(f, '.png'):
            os.rename(f, f.replace('.png', '.jpg'))
        
def is_picture(file):
    exts = ['.jpg', '.png', ".jpeg"]
    return any([ends(file, e) for e in exts])
def is_today(t):
    t = float(t)
    return time2date(t).day == cur_date().day and time2date(t).month == cur_date().month
def is_yesterday(t):
    return time2date(t).day == cur_date().day - 1

def chk_root(root):
    if not ends(root, "/"):
        root += "/"
    return root
def ct():
    return time.time()
def cur_time():
    return time.time()
def cur_date():
    return datetime.datetime.now()
def cur_hour():
    return datetime.datetime.now().hour
def cur_minute():
    return datetime.datetime.now().minute
def cur_second():
    return datetime.datetime.now().second

def eld(st):
    elapsed(st)
def elapsed(st):
    print("Elapsed time: {:.3f}".format(cur_time() - st))

def time2date(t):
    return datetime.datetime.fromtimestamp(t)
def date2time(d):
    try: return int(d.strftime("%s"))
    except: return time.mktime(d.timetuple())
def get_date_string(d):
    return d.strftime("%d.%m.%y %H:%M:%S")
def print_date(d):
    print(get_date_string(d), end = '\n')
def print_time(t, color = None, end = '\n', file=None, year = False):
    form = "%d.%m.%y %H:%M:%S" if year else "%d.%m %H:%M:%S"
    if color is None:
        print(time2date(t).strftime(form), file=file, end = end)
    else:
        print(color.r + time2date(t).strftime(form) + color.e, file=file, end = end)

def form_date_from_time(s):
    current = get_date_string(cur_date())
    dt, tm = current.split(" ")
    tm = s + tm[-3:]
    return dt + " " + tm
def date_to_time_format(string):
    return datetime.datetime.strptime(string, "%d.%m.%y %H:%M:%S")

def zip_dir(path, ziph):
    for root, dirs, files in os.walk(path):
        for f in files:
            ziph.write(os.path.join(root, f))
def zip_files(files, ziph):
    for f in files: ziph.write(f)

def construct_html(elements, filename):
    with open(filename, "w+") as file:
        html_str = "\n\n".join(elements)
        file.write(html_str)

def get_unique_attr(pack, attr):
    output, seen = [], set()
    for t in pack:
        if t[attr] not in seen:
            seen.add(t[attr])
            output.append(t)
    return output

bb = bold_blue
offe = open_ffile
efn = extract_filename
inf = input_file