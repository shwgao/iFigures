import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

figure_size = (20, 8)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 26
BIGGER_SIZE = 26

colors = ['#4793AF', '#FFC470', '#DD5746']


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
        for head_line in f:
            if 'write' in head_line or 'read' in head_line:
                sub_data = {}
                for line in f:
                    if line[0] == '\n':
                        break
                    if line:
                        line = line.split()
                        try:  # if can convert to float
                            sub_data[line[0]] = [float(x) for x in line[1:]]
                        except ValueError:
                            sub_data[line[0]] = line[1:]
                data[head_line.strip()] = sub_data

    return data


# draw bar chart for random write, normalize based on Oracle
def draw_random_write(file='./log_files/motivation.txt'):
    datas = read_data(file)
    data = datas['random write']

    # normalize based on Oracle, save the result in normalized_data
    normalized_data = {}
    for key in data.keys():
        if key != 'label':
            normalized_data[key] = [data[key][i] / data['Oracle'][i] for i in range(len(data[key]))]

    print(normalized_data)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.2
    threshold = 1.2

    index = np.arange(len(data['label']))*0.8

    # draw bars
    i = 0
    for key in normalized_data.keys():
        ax.bar(index+i*bar_width, normalized_data[key], bar_width, label=key, color=colors[i], edgecolor='black')
        i += 1

    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(data['label'], ha='center', fontsize=BIG_SIZE, fontweight='bold')  # rotation=45,

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=1.8, pad=0, sharex=ax)

    # draw bars
    i = 0
    for key in normalized_data.keys():
        axlog.bar(index + i * bar_width, normalized_data[key], bar_width, label=key, color=colors[i],
                  edgecolor='black')
        i += 1

    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 900)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    axlog.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put the value on the top of the bar
    x_place_offset = [[-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.05],
                      [-0.01, -0.01, -0.01, -0.01, -0.04, -0.0, -0.0, -0.0],
                      [0.01, 0.01, 0.01, 0.01, 0.05, 0.04, 0.04, 0.04]]
    for i in range(len(data['label'])):
        for j, key in enumerate(normalized_data.keys()):
            # adjust the text content
            text = data[key][i]
            if text > 1 and text < 100:
                text = str(round(text, 1))
            elif text >= 10:
                text = str(round(text))
            else:
                text = str(round(text, 2))

            if normalized_data[key][i] > threshold:
                axlog.text(index[i] + j * bar_width + x_place_offset[j][i], normalized_data[key][i] * 1.2, text,
                           fontsize=BIG_SIZE, rotation=0,
                           ha='center')
            else:
                ax.text(index[i] + j * bar_width + x_place_offset[j][i], normalized_data[key][i] + 0.01, text,
                        fontsize=BIG_SIZE, rotation=0,
                        ha='center')

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.9)
    axlog.grid(axis='y', linestyle='--', alpha=0.9)

    # set the legend
    axlog.legend(bbox_to_anchor=(0.56, 1), loc='upper left', borderaxespad=0, ncol=3, frameon=True, edgecolor='black',
                 handlelength=1, handletextpad=0.4, fontsize=BIG_SIZE)
    ax.set_ylabel('Normalized Performance', fontsize=BIGGER_SIZE, fontweight='bold', labelpad=20)
    ax.yaxis.set_label_coords(-0.04, 0.65)
    ax.set_xlabel('Block Size', fontsize=BIGGER_SIZE, fontweight='bold')
    ax.set_title('Normalized Random Write Time', fontsize=BIGGER_SIZE, fontweight='bold')
    # plt title
    plt.suptitle('Random Write Micro Benchmark', fontsize=BIGGER_SIZE+3, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/random_write.pdf')


def draw_random_read(file='./log_files/motivation.txt'):
    datas = read_data(file)
    data = datas['random read']

    # normalize based on Oracle, save the result in normalized_data
    normalized_data = {}
    for key in data.keys():
        if key != 'label':
            normalized_data[key] = [data[key][i] / data['Oracle'][i] for i in range(len(data[key]))]

    print(normalized_data)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.2
    threshold = 1.2

    index = np.arange(len(data['label']))*0.8

    # draw bars
    i = 0
    for key in normalized_data.keys():
        ax.bar(index+i*bar_width, normalized_data[key], bar_width, label=key, color=colors[i], edgecolor='black')
        i += 1

    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(data['label'], ha='center', fontsize=BIG_SIZE, fontweight='bold')  # rotation=45,

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=2.0, pad=0, sharex=ax)

    # draw bars
    i = 0
    for key in normalized_data.keys():
        axlog.bar(index + i * bar_width, normalized_data[key], bar_width, label=key, color=colors[i],
                  edgecolor='black')
        i += 1

    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 2000)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    axlog.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put the value on the top of the bar
    x_place_offset = [[-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
                      [-0.01, -0.01, -0.01, -0.01, -0.04, -0.04, -0.06, -0.06],
                      [0.01, 0.01, 0.01, 0.01, 0.05, 0.04, 0.07, 0.07]]
    for i in range(len(data['label'])):
        for j, key in enumerate(normalized_data.keys()):
            # adjust the text content
            text = data[key][i]
            if text >1 and text < 100:
                text = str(round(text, 1))
            elif text >= 10:
                text = str(round(text))
            else:
                text = str(round(text, 2))

            if normalized_data[key][i] > threshold:
                axlog.text(index[i]+j*bar_width+x_place_offset[j][i], normalized_data[key][i]*1.2, text, fontsize=BIG_SIZE, rotation=0,
                           ha='center')
            else:
                ax.text(index[i]+j*bar_width+x_place_offset[j][i], normalized_data[key][i]+0.01, text, fontsize=BIG_SIZE, rotation=0,
                        ha='center')

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.9)
    axlog.grid(axis='y', linestyle='--', alpha=0.9)

    # set the legend
    axlog.legend(bbox_to_anchor=(0.56, 1), loc='upper left', borderaxespad=0, ncol=3, frameon=True, edgecolor='black',
                 handlelength=1, handletextpad=0.4, fontsize=BIG_SIZE)
    ax.set_ylabel('Normalized Performance', fontsize=BIGGER_SIZE, fontweight='bold', labelpad=20)
    ax.yaxis.set_label_coords(-0.05, 0.65)
    ax.set_xlabel('Block Size', fontsize=BIGGER_SIZE, fontweight='bold')
    ax.set_title('Normalized Random Read Time', fontsize=BIGGER_SIZE, fontweight='bold')
    # plt title
    plt.suptitle('Random Read Micro Benchmark', fontsize=BIGGER_SIZE+3, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/random_read.pdf')


def draw_seq_read(file='./log_files/motivation.txt'):
    datas = read_data(file)
    data = datas['sequential read']

    # normalize based on Oracle, save the result in normalized_data
    normalized_data = {}
    for key in data.keys():
        if key != 'label':
            normalized_data[key] = [data[key][i] / data['Oracle'][i] for i in range(len(data[key]))]

    print(normalized_data)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.2
    threshold = 1.2

    index = np.arange(len(data['label']))*0.8

    # draw bars
    i = 0
    for key in normalized_data.keys():
        ax.bar(index+i*bar_width, normalized_data[key], bar_width, label=key, color=colors[i], edgecolor='black')
        i += 1

    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(data['label'], ha='center', fontsize=BIG_SIZE, fontweight='bold')  # rotation=45,

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=2.0, pad=0, sharex=ax)

    # draw bars
    i = 0
    for key in normalized_data.keys():
        axlog.bar(index + i * bar_width, normalized_data[key], bar_width, label=key, color=colors[i],
                  edgecolor='black')
        i += 1

    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 2000)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    axlog.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put the value on the top of the bar
    x_place_offset = [[-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
                      [-0.01, -0.01, -0.01, -0.01, -0.04, -0.04, -0.06, -0.06],
                      [0.01, 0.01, 0.01, 0.01, 0.05, 0.04, 0.07, 0.07]]
    for i in range(len(data['label'])):
        for j, key in enumerate(normalized_data.keys()):
            # adjust the text content
            text = data[key][i]
            if text >1 and text < 100:
                text = str(round(text, 1))
            elif text >= 10:
                text = str(round(text))
            else:
                text = str(round(text, 2))

            if normalized_data[key][i] > threshold:
                axlog.text(index[i]+j*bar_width+x_place_offset[j][i], normalized_data[key][i]*1.2, text, fontsize=BIG_SIZE, rotation=0,
                           ha='center')
            else:
                ax.text(index[i]+j*bar_width+x_place_offset[j][i], normalized_data[key][i]+0.01, text, fontsize=BIG_SIZE, rotation=0,
                        ha='center')

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.9)
    axlog.grid(axis='y', linestyle='--', alpha=0.9)

    # set the legend
    axlog.legend(bbox_to_anchor=(0.56, 1), loc='upper left', borderaxespad=0, ncol=3, frameon=True, edgecolor='black',
                 handlelength=1, handletextpad=0.4, fontsize=BIG_SIZE)
    ax.set_ylabel('Normalized Performance', fontsize=BIGGER_SIZE, fontweight='bold', labelpad=20)
    ax.yaxis.set_label_coords(-0.05, 0.65)
    ax.set_xlabel('Block Size', fontsize=BIGGER_SIZE, fontweight='bold')
    ax.set_title('Normalized Sequential Read Time', fontsize=BIGGER_SIZE, fontweight='bold')
    # plt title
    plt.suptitle('Sequential Read Micro Benchmark', fontsize=BIGGER_SIZE+3, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/sequential_read.pdf')


def draw_seq_write(file='./log_files/motivation.txt'):
    datas = read_data(file)
    data = datas['sequential write']

    # normalize based on Oracle, save the result in normalized_data
    normalized_data = {}
    for key in data.keys():
        if key != 'label':
            normalized_data[key] = [data[key][i] / data['Oracle'][i] for i in range(len(data[key]))]

    print(normalized_data)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.2
    threshold = 1.2

    index = np.arange(len(data['label']))*0.8

    # draw bars
    i = 0
    for key in normalized_data.keys():
        ax.bar(index+i*bar_width, normalized_data[key], bar_width, label=key, color=colors[i], edgecolor='black')
        i += 1

    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(data['label'], ha='center', fontsize=BIG_SIZE, fontweight='bold')  # rotation=45,

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=2.0, pad=0, sharex=ax)

    # draw bars
    i = 0
    for key in normalized_data.keys():
        axlog.bar(index + i * bar_width, normalized_data[key], bar_width, label=key, color=colors[i],
                  edgecolor='black')
        i += 1

    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 2000)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    axlog.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put the value on the top of the bar
    x_place_offset = [[-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
                      [-0.01, -0.01, -0.01, -0.01, -0.04, -0.04, -0.06, -0.06],
                      [0.01, 0.01, 0.01, 0.01, 0.05, 0.04, 0.07, 0.07]]
    for i in range(len(data['label'])):
        for j, key in enumerate(normalized_data.keys()):
            # adjust the text content
            text = data[key][i]
            if text >1 and text < 100:
                text = str(round(text, 1))
            elif text >= 10:
                text = str(round(text))
            else:
                text = str(round(text, 2))

            if normalized_data[key][i] > threshold:
                axlog.text(index[i]+j*bar_width+x_place_offset[j][i], normalized_data[key][i]*1.2, text, fontsize=BIG_SIZE, rotation=0,
                           ha='center')
            else:
                ax.text(index[i]+j*bar_width+x_place_offset[j][i], normalized_data[key][i]+0.01, text, fontsize=BIG_SIZE, rotation=0,
                        ha='center')

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.9, zorder=0)
    axlog.grid(axis='y', linestyle='--', alpha=0.9)

    # set the legend
    axlog.legend(bbox_to_anchor=(0.56, 1), loc='upper left', borderaxespad=0, ncol=3, frameon=True, edgecolor='black',
                 handlelength=1, handletextpad=0.4, fontsize=BIG_SIZE)
    ax.set_ylabel('Normalized Performance', fontsize=BIGGER_SIZE, fontweight='bold', labelpad=20)
    ax.yaxis.set_label_coords(-0.05, 0.65)
    ax.set_xlabel('Block Size', fontsize=BIGGER_SIZE, fontweight='bold')
    ax.set_title('Normalized Sequential Write Time', fontsize=BIGGER_SIZE, fontweight='bold')
    # plt title
    plt.suptitle('Sequential Write Micro Benchmark', fontsize=BIGGER_SIZE+3, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/sequential_write.pdf')


if __name__ == '__main__':
    draw_random_write()
    # draw_random_read()
    # draw_seq_read()
    # draw_seq_write()
