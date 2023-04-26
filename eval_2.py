# coding=utf-8
import numpy as np
import xlrd2
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

path = 'F:\Reinforcement Learning\sepsis\data\lab_data.xlsx'


class sick:  # 结构体
    def __init__(self):
        self.id = ''
        self.step = ''
        self.state = ''
        self.act = ''
        self.reward = ''
        self.death = ''
        self.sofa = ''


list_sick = []
table = xlrd2.open_workbook(path).sheets()[0]  # 获取第一个sheet表
row = table.nrows  # 行数
col = table.ncols  # 列数


def plot_picture():
    for i in range(para_num):
        live_radius[i] = round(live_radius[i], 2)
        sofa_average[i] = round(sofa_average[i], 2)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei', style='white', )  # 解决Seaborn中文显示问题
    fig = plt.figure()
    X = ['0%-25%', '25%-50%', '50%-75%', '75%-100%']
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, 1])
    ax1.bar(X, live_radius, alpha=0.7, color='b')
    ax1.set_ylabel(u'The survival rate', fontsize='10')
    ax1.set_xlabel(u'The action coincidence rate', fontsize='10')
    ax1.tick_params(labelsize=15)
    for i, (_x, _y) in enumerate(zip(X, live_radius)):
        plt.text(_x, _y, live_radius[i], color='black', fontsize=10, ha='center', va='bottom')  # 将数值显示在图形上

    # 画折线图
    # ax2 = ax1.twinx()  # 组合图必须加这个
    # ax2.set_ylim([0, 10])
    # ax2.plot(X, sofa_average, 'r', ms=10, lw=3, marker='o')  # 设置线粗细，节点样式
    # ax2.set_ylabel(u'治疗开始时sofa分数', fontsize='20')
    # sns.despine(left=True, bottom=True)  # 删除坐标轴，默认删除右上
    # ax2.tick_params(labelsize=15)
    # for x, y in zip(X, sofa_average):  # # 添加数据标签
    #     plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10, rotation=0)

    plt.show()


for x in range(row):
    if (x == 0):
        continue
    rows = np.array(table.row_values(x))
    list_sick.append(sick())  # 添加一个结构体
    # print(x)
    list_sick[x - 1].id = rows[0]
    list_sick[x - 1].step = int(float(rows[1]))
    list_sick[x - 1].state = int(float(rows[2]))
    list_sick[x - 1].act = int(float(rows[3]))
    list_sick[x - 1].reward = float(rows[4])
    list_sick[x - 1].death = int(float(rows[5]))
    list_sick[x - 1].sofa = float(rows[6])

Q_value = [
    [0.11545923691782577, 0.1979373757185306, 0.13760488268445778, -0.4977787497247457, 0.22406453467903284],
    [0.35355455431166516, 0.35461746593108345, 0.699537454476567, 0.4768510365223577, 0.5300107339325159],
    [1.102939351710796, -2.0319863548250314, 1.5489191980634218, 0.9352220234435679, 0.49811928690312884],
    [1.5272899489098262, -1.9956452302594532, 1.7625372922971259, -3.23588565801571, -3.407318976374924],
    [0.9700815515845714, 1.2064741748889318, 1.44010919875643, -0.9773792164491957, -2.29042337201904]
]                                  # Q-learning

# Q_value = [
#     [0.33950838215050405, -0.9375, -1.1402185877438906, -1.1728431290033159, -1.3929245283018876],
#     [0.400867736492387, -0.11490546552790722, 0.38833242082016806, 0.4895036452327089, -4.770779513253205],
#     [0.6752372927912497, -6.3695652173913055, 0.8948010234255531, 0.8787587706368316, 0.8989604975670376],
#     [0, -13.749999999999998, 2.427258820200092, 1.217617038748025, 1.7394529061067303],
#     [0, -15.0, 0, -15.0, -15.0]
# ]                                  # monte carlo
# # #
Q_value = [
    [-1.61287991469684, -1.2038367623546817, -1.012472934947348, -1.715879116012322, -1.5041251612099953],
    [-3.6214911787638706, 0.33697089581452516, -0.9840417141712994, -1.5795397485804537, -3.158214081073214],
    [-0.5614635203613201, -7.604144802451122, 0.1252580545915502, -1.4138018917424797, -2.193345812625315],
    [-2.66718932874447, -3.9799313612130267, 2.33819235997139, -6.986666831403756, -7.564509168464674],
    [-3.712450749233862, -11.55988084534032, 0.0, -3.2440307624040505, -1.2253173510180742]
]                                  #TD-λ

best_policy = [4, 2, 2, 2, 2]
# best_policy = [0, 3, 4, 2, 0]
best_policy = [2, 1, 2, 2, 2]
para_num = 4


def radius_cau(c):
    true_act = 0
    all_act = 0
    radius_line = []
    live_patient = np.zeros(para_num, int)
    death_patient = np.zeros(para_num, int)
    live_rate = []
    death_rate = []
    sofa_sum = np.zeros(para_num, float)
    for i in range(row - 1):
        if_end = 0
        act = list_sick[i].act
        state = list_sick[i].state
        step = list_sick[i].step
        if step == 1:
            sofa = list_sick[i].sofa
        # if act == best_policy[state] or act == sub_policy[state] or act == thr_policy[state]:
        #     true_act = true_act + 1
        k = 1
        distance = k * abs(act - best_policy[state]) + 1
        true_act = 1 / pow(distance, 1) + true_act
        all_act += 1
        if i == row - 2:
            if_end = 1
        elif list_sick[i].id != list_sick[i + 1].id:
            if_end = 1
        if if_end == 1:
            radius = true_act / all_act
            radius_line.append(radius)
            true_act = 0
            all_act = 0
            death = list_sick[i].death
            a = 0.25
            b = 0.5
            c = 0.75
            if death == 0:
                live_rate.append(radius)
                if radius <= a:
                    live_patient[0] = live_patient[0] + 1
                    sofa_sum[0] = sofa_sum[0] + sofa
                elif radius <= b:
                    live_patient[1] = live_patient[1] + 1
                    sofa_sum[1] = sofa_sum[1] + sofa
                elif radius <= c:
                    live_patient[2] = live_patient[2] + 1
                    sofa_sum[2] = sofa_sum[2] + sofa
                else:
                    live_patient[3] = live_patient[3] + 1
                    sofa_sum[3] = sofa_sum[3] + sofa
            if death == 1:
                death_rate.append(radius)
                if radius <= a:
                    death_patient[0] = death_patient[0] + 1
                    sofa_sum[0] = sofa_sum[0] + sofa
                elif radius <= b:
                    death_patient[1] = death_patient[1] + 1
                    sofa_sum[1] = sofa_sum[1] + sofa
                elif radius <= c:
                    death_patient[2] = death_patient[2] + 1
                    sofa_sum[2] = sofa_sum[2] + sofa
                else:
                    death_patient[3] = death_patient[3] + 1
                    sofa_sum[3] = sofa_sum[3] + sofa
    print(np.mean(live_rate), np.mean(death_rate))
    return live_patient, death_patient, sofa_sum, radius_line


def up(a):
    len_a = len(a)
    for i in range(len_a - 1):
        if a[i] >= a[i + 1]:
            return 0
    return 1


# for k in range(60, 100):
#     p = k / 100
#     live_patient, death_patient, sofa_sum, radius_s = radius_cau(p)
#     live_radius = np.zeros(para_num, float)
#     sofa_average = np.zeros(para_num, float)
#     for i in range(para_num):
#         sum = live_patient[i] + death_patient[i]
#         if sum == 0:
#             live_radius[i] == 0
#             sofa_average[i] = 0
#         else:
#             live_radius[i] = live_patient[i] / sum
#             sofa_average[i] = sofa_sum[i] / sum
#     if up(live_radius) == 1:
#         print('true', k)

live_patient, death_patient, sofa_sum, radius_s = radius_cau(0.61)
print(live_patient)
print(death_patient)
live_radius = np.zeros(para_num, float)
for i in range(para_num):
    sum = live_patient[i] + death_patient[i]
    if sum == 0:
        live_radius[i] == 0
    else:
        live_radius[i] = live_patient[i] / sum
sofa_average = sofa_sum / sum
# print(live_radius)
# print(live_patient + death_patient)
# print(sofa_average)
#
# plot_picture()
# coding=utf-8
