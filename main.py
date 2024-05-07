
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

figure_size = (20, 8)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 26
BIGGER_SIZE = 26


def read_data(file='./log_files/motivation.txt'):
    """ motivation.txt
    random write
    label        1KB     2KB     4KB     8KB     32KB     512KB   2MB      4MB
    Oracle      0.03    0.06    0.17    0.30    1.28     132.0   530.0    1058.0
    Ethernet    4.54    5.09    6.13    8.11    21.35    304.51  1204.68  2404.6
    Infiniband  2.57    2.85    3.27    4.03    8.42     123.2   482.95   968.81
    """
    # read data to a dictionary
    data = {}
    with open(file, 'r') as f:
        for line in f:
            if 'random write' in line:
                for line in f:
                    if line:
                        line = line.split()
                        try:  # if can convert to float
                            data[line[0]] = [float(x) for x in line[1:]]
                        except ValueError:
                            data[line[0]] = line[1:]

    return data


# draw bar chart for random write, normalize based on Oracle
def draw_random_write(file='./log_files/motivation.txt'):
    data = read_data(file)

    # normalize based on Oracle, save the result in normalized_data
    normalized_data = {}
    for key in data.keys():
        if key != 'label':
            normalized_data[key] = [data[key][i] / data['Oracle'][i] for i in range(len(data[key]))]

    print(normalized_data)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.2
    threshold = 1.2

    index = np.arange(len(data['label']))

    # draw bars
    i = 0
    for key in normalized_data.keys():
        ax.bar(index+i*bar_width, normalized_data[key], bar_width, label=key)
        i += 1

    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(data['label'], ha='right', fontsize=BIG_SIZE, fontweight='bold')  # rotation=45,

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=1.5, pad=0, sharex=ax)

    # draw bars
    i = 0
    for key in normalized_data.keys():
        axlog.bar(index + i * bar_width, normalized_data[key], bar_width, label=key)
        i += 1

    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 300)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    axlog.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put the value on the top of the bar
    for i in range(len(data['label'])):
        for key in normalized_data.keys():
            ax.text(i + 0.1, normalized_data[key][i] + 0.1, str(round(normalized_data[key][i], 2)), fontsize=SMALL_SIZE)

    # set the legend
    # axlog.legend(loc='upper right', fontsize=BIG_SIZE)
    axlog.legend(bbox_to_anchor=(0.5, 1.12), loc='upper left', borderaxespad=0, ncol=3, frameon=False,
                 handlelength=1, handletextpad=0.4, fontsize=BIG_SIZE)
    ax.set_ylabel('Normalized Time', fontsize=BIGGER_SIZE, fontweight='bold')
    ax.set_xlabel('Block Size', fontsize=BIGGER_SIZE, fontweight='bold')
    ax.set_title('Normalized Random Write Time', fontsize=BIGGER_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    draw_random_write()
