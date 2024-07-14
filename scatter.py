import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
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


Science_language = [[2015, 2017, 2019, 2020, 2021, 2022, 2023], [10, 110, 110, 406, 700, 1300, 22000]]
Biology = [[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], [30, 10, 41, 100, 355, 770, 34000, 70000]]
Geography = [[2015, 2016, 2017, 2019, 2021, 2022, 2024], [10, 60, 40, 41, 82, 110, 30000]]
Climate = [[2016, 2017, 2018, 2019, 2020, 2021, 2023, 2024], [10, 10, 20, 25, 108, 200, 500, 1300]]

data = {'Science_language': Science_language, 'Biology': Biology, 'Geography': Geography, 'Climate': Climate}

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
                           bbox_to_anchor=(1.02, 0.45), title_fontsize='large', fontsize='large'
                           , borderaxespad=0., ncol=1, frameon=False, labelspacing=1.1)

# 添加第二个图例到图中
plt.gca().add_artist(second_legend)

# Customize legend
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), scatterpoints=1, title="Domain", title_fontsize='large', loc='upper left', fontsize='large', 
           markerscale=0.5, bbox_to_anchor=(1., 0.95), borderaxespad=0., ncol=1, frameon=False)

plt.xlabel("Year", fontsize=20)
plt.ylabel("Parameters (in Millions)", fontsize=20)

#set ylimit to 100
plt.ylim(7, 120000)

# set ticks fontsize
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


plt.tight_layout()
plt.savefig('scatter.png')
