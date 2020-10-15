import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

def plot_compare_pack_vars(evolutions, colors=[], legend_titles=[], loglog=0,
                           evolutions_vars=[], colors_var=[]):
    assert len(evolutions) == len(evolutions_vars)
    plt.figure(figsize=(12, 6))
    plt.title(title)

    pack = common_shorten(pack)
    for line, c in zip(pack, colors):
        method = plt.semilogy if log_scale else plt.plot
        method = plt.loglog if loglog else method
        method(line, lw=1.3, c=c)
    plt.xlabel("# iteration")
    plt.ylabel(r"$\log\left( f/f* - 1\right)$ ")
    plt.rcParams['grid.alpha'] = 0.4
    plt.grid(1)
    plt.legend(handles=[mpatches.Patch(color=c, label=lt) for c, lt in zip(colors, legend_titles)])
    plt.show()



def plot_result(result, title="Algorithm result", log_scale=True, loglog=False, name=''):
    plt.figure(figsize=(12, 6))
    plt.title(title)

    method = plt.semilogy if log_scale else plt.plot
    method = plt.loglog if loglog else method
    method(result, lw=1.3)

    plt.rcParams['grid.alpha'] = 0.4
    plt.grid(1)
    
    n_exp = read_int('other/n_exp.txt')
    plt.savefig("figures/" + name + "---" + str(n_exp) + '.png')
    store_int(n_exp + 1, 'other/n_exp.txt')
    
    plt.show()


def plot_compare_pack(pack, colors, title="Comparison", log_scale=True, 
                        legend_titles=[], loglog=False, name=''):
    plt.figure(figsize=(12, 6))
    plt.title(title)

    pack = common_shorten(pack)
    for line, c in zip(pack, colors):
        method = plt.semilogy if log_scale else plt.plot
        method = plt.loglog if loglog else method
        method(line, lw=1.3, c=c)
    plt.xlabel("# iteration")
    plt.ylabel(r"$\log\left( f/f* - 1\right)$ ")
    plt.rcParams['grid.alpha'] = 0.4
    plt.grid(1)
    plt.legend(handles=[mpatches.Patch(color=c, label=lt) for c, lt in zip(colors, legend_titles)])
    n_exp = read_int('other/n_exp.txt')
    plt.savefig("figures/" + name + "---" + str(n_exp) + '.png')
    store_int(n_exp + 1, 'other/n_exp.txt')
    plt.show()


def plot_compare(red, blue, title="Comparison", log_scale=True, 
                        legend_titles=[], loglog=False, name=''):
    assert len(legend_titles) == 2
    plt.figure(figsize=(12, 6))
    plt.title(title)
    method = plt.semilogy if log_scale else plt.plot
    method = plt.loglog if loglog else method
    method(blue, lw=1.3, c='b')
    method(red, lw=1.3, c='r')
    plt.xlabel("# iteration")
    plt.ylabel(r"$\log\left( f/f* - 1\right)$ ")
    red_patch = mpatches.Patch(color='red', label=legend_titles[0])
    blue_patch = mpatches.Patch(color='blue', label=legend_titles[1])
    plt.legend(handles=[red_patch, blue_patch])

    plt.rcParams['grid.alpha'] = 0.4
    plt.grid(1)
    plt.show()


def solution_flow(circles, triangs):
    ext, grid_num = 15, 150
    start, ended = 0, 220

    x_min = min(min(triangs[start:ended, 0]), min(circles[start:ended, 0]))[0] - 1
    x_max = max(max(triangs[start:ended, 0]), max(circles[start:ended, 0]))[0] + 1
    y_min = min(min(triangs[start:ended, 1]), min(circles[start:ended, 1]))[0] - 1
    y_max = max(max(triangs[start:ended, 1]), max(circles[start:ended, 1]))[0] + 1

    x, y = np.linspace(x_min, x_max, num=grid_num), np.linspace(y_min, y_max, num=grid_num)
    x, y = np.meshgrid(x, y)  # get 2d matrices covering the grid
    z = function.eval_2d(x, y)
    z -= z.min()

    # PLOTTING STUFF
    fig, ax = plt.subplots(figsize=(16, 8))
    p = ax.pcolor(x, y, z, cmap=cm.BuPu, norm=colors.PowerNorm(0.3), alpha=0.7)
    cb = fig.colorbar(p)
    col = cm.RdPu(np.linspace(0, 1, ended - start))
    color = cm.Blues(np.linspace(0, 1, ended - start))
    ax.scatter(triangs[start:ended, 0], triangs[start:ended, 1], marker='^', c=col, linewidths=0.5)
    ax.scatter(circles[start:ended, 0], circles[start:ended, 1], marker='o', c=color, linewidths=0.5)

def common_shorten(sequences, optimum=None, mask=None):
    common_len = min([len(l) for l in sequences])
    if optimum is not None:
        return tuple(convert(l[:common_len], optimum) if m else l[:common_len] for (l, m) in zip(sequences, mask))
    else:
        return tuple(l[:common_len] for l in sequences)

def read_int(filename):
    return int(input_file(filename)[0])
def store_int(i, filename, regime='w'):
    print(i, file=open(filename, regime), end='\n')
def input_file(filename, encoding="utf-8", to_strip=True, ignore=0):
    errors = 'ignore' if ignore else None
    if to_strip:
        return [l.strip() for l in open(filename, encoding=encoding, errors=errors)]
    else:
        return [l for l in open(filename, encoding=encoding, errors=errors)]