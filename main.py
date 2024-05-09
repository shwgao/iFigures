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
            if len(head_line.split()) <= 2:
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


def draw_cg_perform():
    datas = read_data('./log_files/motivation.txt')
    data = datas['CG performance']

    line_data = [x/(1024**2) for x in data['PeakMem']]
    bar_data = [x/60 for x in data['Exetime']]

    x_ticklabels = data['label']
    title = 'CG'

    line_label = 'Peak Memory Usage (GB)'
    bar_label = 'Execution Time (min)'
    xlabel = ''

    bar_scale = 'linear'
    line_scale = 'linear'

    bar_color = colors[0]
    line_color = colors[1]

    file_name = './figures/CG_performance.pdf'

    y_tickles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]

    draw_double_axis(bar_data, line_data, bar_label, line_label, title, xlabel, x_ticklabels, file_name, bar_color,
                     line_color, bar_scale, line_scale, fig_size=(13, 8), font_size=BIGGER_SIZE, y_tickles=y_tickles)


def draw_double_axis(bar_data, line_data, bar_label, line_label, title, xlabel, x_ticklabels, file_name, bar_color,
                     line_color, bar_scale='linear', line_scale='linear', fig_size=(20, 8), font_size=BIGGER_SIZE,
                     y_tickles=None):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax2 = ax.twinx()

    index = np.arange(len(bar_data))*0.5

    # 绘制柱状图
    bar_width = 0.3
    ax.bar(index, bar_data, bar_width, color=bar_color, edgecolor='black')
    ax.set_ylabel(bar_label, fontsize=font_size, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight='bold')
    ax.set_title(title, fontsize=font_size+3, fontweight='bold')

    # 绘制折线图
    ax2.plot(index, line_data, color=line_color, marker='o', markersize=15, label=line_label, linewidth=4)
    ax2.set_ylabel(line_label, fontsize=font_size, fontweight='bold')

    # 调整x轴刻度标签
    ax.set_xticks(index)
    ax.set_xticklabels(x_ticklabels, fontsize=font_size, fontweight='bold')

    # set y ticks' format
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    ax2.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # set scale
    ax.set_yscale(bar_scale)
    ax2.set_yscale(line_scale)

    # 调整y轴范围
    ax.set_ylim(0, 225)
    ax2.set_ylim(0, 9.3)
    if y_tickles:
        ax.set_yticks()
    # ax.autoscale()
    # ax2.autoscale()

    # put the number near the line
    for i in range(len(line_data)):
        ax2.text(index[i], line_data[i], round(line_data[i], 1), ha='center', va='bottom', fontsize=BIG_SIZE)

    bar_proxy = plt.Rectangle((0, 0), 1, 1, fc=bar_color)
    line_proxy = plt.Line2D([0], [0], color=line_color, lw=2)
    ax.legend(handles=[bar_proxy, line_proxy], labels=[bar_label, line_label], loc='upper right', fontsize=BIG_SIZE,
              edgecolor='black', frameon=True, handlelength=1, handletextpad=0.4, bbox_to_anchor=(0.8, 1.02))

    ax.grid(axis='y', linestyle='--', alpha=0.9, zorder=0)

    plt.tight_layout()

    # 保存图片
    plt.savefig(file_name)
    # plt.close()
    plt.show()


def draw_NP_1():
    labels = ['Init', '1', '2', '3', '4', '5']
    data1 = [142014098, 0, 0, 0, 0, 258]
    data2 = [108, 2, 0, 0, 1, 109]
    data3 = [93.53, 84.37, 84.37, 84.37, 84.37, 88.56, 95.23, 0.27]
    data = [data1, data2, data3]
    file_names = ['./figures/NP_1_1.pdf', './figures/NP_1_2.pdf', './figures/NP_1_3.pdf']
    colors = ['#4793AF', '#4793AF', '#FFC470']

    # draw three bars separately
    for i in range(3):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        labels = ['Init', '1', '2', '3', '4', '5', 'Do>4KB', 'Do<4KB'] if i == 2 else labels

        ax.bar(labels, data[i], color=colors[i], edgecolor='black')

        ax.set_ylabel('Number of Data Object', fontsize=BIGGER_SIZE, fontweight='bold')
        ax.set_xticklabels(labels, fontsize=BIGGER_SIZE, fontweight='bold')
        ax.yaxis.set_tick_params(labelsize=BIGGER_SIZE)
        ax.set_xlabel('Iteration', fontsize=BIGGER_SIZE, fontweight='bold')

        if i == 0:
            ax.set_title('Data Object Size <= 4KB', fontsize=BIGGER_SIZE + 3, fontweight='bold')
            ax.autoscale()
            ax.set_yscale('log')
            ax.set_ylim(0, 4e8)
            plt.subplots_adjust(left=0.15)
            place = [j if j > 200 else 150 for j in data[i]]
            text = ['1.4e+08', '0', '0', '0', '0', '258']
        elif i == 1:
            ax.set_title('Data Object Size >= 4KB', fontsize=BIGGER_SIZE + 3, fontweight='bold')
            ax.set_yscale('log')
            ax.set_ylim(0, 2e2)

            plt.subplots_adjust(left=0.15)  # 增加left的值
            place = [j if j >= 1 else 0.8 for j in data[i]]
            text = data[i]
        else:
            ax.set_title('Peak Memory Usage', fontsize=BIGGER_SIZE, fontweight='bold')
            ax.set_ylabel('Peak Memory Usage(GB)', fontsize=BIGGER_SIZE, fontweight='bold')
            # set ax[0]'s y-axis scientific notation
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # ax.tick_params(axis='y', labelsize=BIGGER_SIZE)
            text = [str(round(j)) for j in data[i]]
            text[-1] = '27MB'
            ax.set_ylim(0, 110)
            place = data[i]

            labels = ax.get_xticklabels()
            labels[-2].set_fontsize(BIGGER_SIZE-8)
            labels[-1].set_fontsize(BIGGER_SIZE-8)
            labels[-2].set_rotation(15)
            labels[-1].set_rotation(15)
            ax.set_xticklabels(labels)

        # put the number on the top of the bar
        for j in range(len(labels)):
            # if number greater than 1 Million, use scientific notation
            ax.text(j, place[j], text[j], ha='center', va='bottom', fontsize=BIG_SIZE)

        ax.grid(axis='y', linestyle='--', alpha=0.9, zorder=0)

        plt.tight_layout()

        plt.savefig(file_names[i])
        plt.show()


if __name__ == '__main__':
    # draw_random_write()
    # draw_random_read()
    # draw_seq_read()
    # draw_seq_write()
    # draw_cg_perform()
    draw_NP_1()
