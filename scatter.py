import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

# mathmatics = [[2021, 2022, 2023, 2024], [0.34, 2.7, 70, 70]]
# Science_language = [[2019, 2020, 2021, 2022, 2023], [0.11, 0.4, 0.7, 1.3, 22]]
# Physics = [[2024], [7]]
# Biology = [[2020, 2021, 2022, 2023], [0.36, 0.77, 34, 70]]
# Geography = [[2021, 2024], [0.82, 30]]
# Chemistry = [[2022, 2024], [0.77, 13]]

# data = {'Mathmatics': mathmatics, 'Science_language': Science_language, 'Physics': Physics, 
#         'Biology': Biology, 'Geography': Geography, 'Chemistry': Chemistry}


# Projection = [[2022, 2023, 2024], [0.36, 36.7, 300]]
# Downscaling = [[2021, 2022, 2023, 2024], [50, 108, 110, 500]]
# Global_forecasting = [[2021, 2022, 2023, 2023, 2024], [30, 100, 50, 110, 1300]]
# Seasonal_forecasting = [[2021, 2022, 2023, 2024], [2.7, 10, 50, 100]]

# data = {'Projection': Projection, 'Downscaling': Downscaling, 'Global_forecasting': Global_forecasting,
#         'Seasonal_forecasting': Seasonal_forecasting}


# Science_language = [[2015, 2017, 2019, 2020, 2021, 2022, 2023], [10, 110, 110, 406, 700, 1300, 22000]]
# Material = [[2015, 2017, 2017, 2020, 2021], [1, 2, 3.2, 1000, 112], ['OQMD', 'ChEMBLE', 'PubChemQC', 'ZINC20', 'PubChem']]
# Biology = [[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], [30, 10, 41, 100, 355, 770, 34000, 70000], ['Alphafold2']]
# Geography = [[2015, 2016, 2017, 2019, 2021, 2022, 2024], [10, 60, 40, 41, 82, 110, 30000]]
# Climate = [[2016, 2017, 2018, 2019, 2020, 2021, 2023, 2024], [10, 10, 20, 25, 108, 200, 500, 1300]]

Material = [[2015, 2017, 2017, 2020, 2021], [1, 2, 3.2, 1000, 112], ['OQMD', 'ChEMBLE', 'PubChemQC', 'ZINC20', 'PubChem']]
Biology = [[2020, 2021, 2022, 2023, 2024], [93, 650, 738, 4000, 9000], ['Alphafold2', 'ESM-1v', 'ProtGPT2', 'ProGen2', 'ProLLaMA']]
Geography = [[2022, 2023, 2024], [80, 7000, 30000], ['SpaBERT', 'K2', 'GeoGalactica']]
Climate = [[2016, 2017, 2018, 2022, 2023, 2023, 2024, 2023], [1, 66, 10, 37, 82, 110, 6000, 1000000], ['LSTM', 'ExtremeWeather', 'ConvLSTM', 'GraphCast', 'ClimateBERT', 'ClimaX', 'OceanGPT', 'PanGu-Σ']]

data = {'Material': Material, 'Biology': Biology, 'Geography': Geography, 'Climate': Climate}

# colors = {'Mathmatics':'red', 'Science_language':'blue', 'Physics':'green', 
#           'Biology':'yellow', 'Geography':'purple', 'Chemistry':'orange'}
# Use seaborn color palette
palette = sns.color_palette("Set2", len(data))

colors = {key: palette[i] for i, key in enumerate(data.keys())}

plt.figure(figsize=(10, 6))

label = list(data.keys())

for i in range(len(data)):
    year = data[label[i]][0]
    parameters = data[label[i]][1]
    
    plt.scatter(year, parameters, color=colors[label[i]], label=label[i], s=[200*np.log(p+1) for p in parameters])

# put text on the points
offset = [[[-0.2, 0.1], [0.5, -1], [0, 0.5], [0, 1000], [0, 80]], 
          [[-1, -60], [0, 500], [1, 0], [0, -2500], [0, 1000]], 
          [[0, 30], [0, 10000], [-0.3, 40000]], 
          [[0, 0.1],  [-0.2, 30], [0, 5], [0, -23], [0.1, -50], [0, 75], [0, -1000], [0, 0]]]
for i in range(len(data)):
    year = data[label[i]][0]
    parameters = data[label[i]][1]
    labels = data[label[i]][2]
    ofs = offset[i]
    s=[200*np.log(p+1)//2 for p in parameters]
    for j in range(len(year)):
        plt.text(year[j]+ofs[j][0], parameters[j]+ofs[j][1], labels[j], fontsize=16, horizontalalignment='center', 
                 verticalalignment='bottom', color='black', weight='bold')

# log scale y-axis
plt.yscale('log')

plt.grid(True, which="both", ls="--")

# 添加新的图例来表示圈的大小
sizes = [10, 100, 1000, 10000]  # 选择一些代表性的大小
legend_elements = [Line2D([0], [0], marker='o', color='black', label=f'{size}M',
                          markerfacecolor='gray', markersize=np.sqrt(200*np.log(size+1)))
                   for size in sizes]

# 创建第二个图例
second_legend = plt.legend(handles=legend_elements, title="Parameters", loc='upper left', 
                           bbox_to_anchor=(1.02, 0.45), title_fontsize=18, fontsize=15
                           , borderaxespad=0., ncol=1, frameon=False, labelspacing=1.1)
second_legend.get_title().set_fontweight('bold')

# 添加第二个图例到图中
plt.gca().add_artist(second_legend)

# Customize legend
from matplotlib.lines import Line2D

# 在绘制散点图之后，创建图例之前

# 创建自定义图例元素
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10)
                   for label, color in colors.items()]

# 创建自定义图例
legend = plt.legend(handles=legend_elements, title="Domain", title_fontsize=18, markerscale=1.5, 
                    loc='upper left', fontsize=15, bbox_to_anchor=(1., 0.95), 
                    borderaxespad=0., ncol=1, frameon=False)
legend.get_title().set_fontweight('bold')

# handles, labels = plt.gca().get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# plt.legend(unique_labels.values(), unique_labels.keys(), scatterpoints=1, title="Domain", title_fontsize='large', loc='upper left', fontsize='large', 
#            markerscale=0.5, bbox_to_anchor=(1., 0.95), borderaxespad=0., ncol=1, frameon=False)

# plt.xlabel("Year", fontsize=20)
plt.ylabel("Parameters (in Millions)", fontsize=20, weight='bold')

#set ylimit to 100
plt.xlim(2014, 2024.9)
plt.ylim(0.6, 3500000)

# set ticks fontsize
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


plt.tight_layout()
plt.savefig('scatter.pdf')
