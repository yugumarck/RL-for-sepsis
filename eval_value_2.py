# coding=utf-8
# coding=utf-8
import numpy as np
import xlrd2
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_picture(p_num, radius):
    for i in range(p_num):
            radius[i] = round(radius[i], 2)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei', style='white', )  # 解决Seaborn中文显示问题
    fig = plt.figure()
    X = ['-11~-8', '-8~-5', '-5~-2', '-2~1']
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, 1])
    ax1.bar(X, radius, alpha=0.7, color='b')
    ax1.set_ylabel(u'The survival rate', fontsize='10')
    ax1.set_xlabel(u'The relative gain', fontsize='10')
    ax1.tick_params(labelsize=15)
    for i, (_x, _y) in enumerate(zip(X, radius)):
        plt.text(_x, _y, radius[i], color='black', fontsize=10, ha='center', va='bottom')  # 将数值显示在图形上
    plt.show()


list_sick = []
table = xlrd2.open_workbook(path).sheets()[0]  # 获取第一个sheet表
row = table.nrows  # 行数
col = table.ncols  # 列数

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

# para_num = 3
# live_patient = np.zeros(shape=(2, para_num))
# death_patient = np.zeros(shape=(2, para_num))

Q_value = [
    [0.11545923691782577, 0.1979373757185306, 0.13760488268445778, -0.4977787497247457, 0.22406453467903284],
    [0.35355455431166516, 0.35461746593108345, 0.699537454476567, 0.4768510365223577, 0.5300107339325159],
    [1.102939351710796, -2.0319863548250314, 1.5489191980634218, 0.9352220234435679, 0.49811928690312884],
    [1.5272899489098262, -1.9956452302594532, 1.7625372922971259, -3.23588565801571, -3.407318976374924],
    [0.9700815515845714, 1.2064741748889318, 1.44010919875643, -0.9773792164491957, -2.29042337201904]
]                                  # Q-learning

Q_value = [
    [0.33950838215050405, -0.9375, -1.1402185877438906, -1.1728431290033159, -1.3929245283018876],
    [0.400867736492387, -0.11490546552790722, 0.38833242082016806, 0.4895036452327089, -4.770779513253205],
    [0.6752372927912497, -6.3695652173913055, 0.8948010234255531, 0.8787587706368316, 0.8989604975670376],
    [0, -13.749999999999998, 2.427258820200092, 1.217617038748025, 1.7394529061067303],
    [0, -15.0, 0, -15.0, -15.0]
]                                  # monte carlo
#
# Q_value = [
#     [-1.61287991469684, -1.2038367623546817, -1.012472934947348, -1.715879116012322, -1.5041251612099953],
#     [-3.6214911787638706, 0.33697089581452516, -0.9840417141712994, -1.5795397485804537, -3.158214081073214],
#     [-0.5614635203613201, -7.604144802451122, 0.1252580545915502, -1.4138018917424797, -2.193345812625315],
#     [-2.66718932874447, -3.9799313612130267, 2.33819235997139, -6.986666831403756, -7.564509168464674],
#     [-3.712450749233862, -11.55988084534032, 0.0, -3.2440307624040505, -1.2253173510180742]
# ]                                  #TD-λ

# best_policy = [4, 2, 2, 2, 2]
# worst_policy = [3, 0, 1, 4, 4]
best_policy = [0, 3, 4, 2, 0]
worst_policy = [4, 4, 1, 1, 1]
# best_policy = [2, 1, 2, 2, 2]
# worst_policy = [3, 0, 1, 4, 1]

true_act = 0
false_act = 0
# sofa_sum = np.zeros(shape=(2, para_num))

def value_cau(min_r=-10000, max_r=10000):
    death_line = []
    r_line = []
    s_line_live = []
    s_line_death = []
    state_line = []
    for i in range(row - 1):
        if_end = 0
        if_start = 0
        act = list_sick[i].act
        state = list_sick[i].state
        death = list_sick[i].death
        if i == 0:
            if_start = 1
        elif list_sick[i].id != list_sick[i - 1].id:
            if_start = 1
        if if_start == 1:
            r = 0
            s = 0
        if Q_value[state][best_policy[state]] != 0:
            r = r + Q_value[state][act] / Q_value[state][best_policy[state]]
        else:
            r = r
        if Q_value[state][act] > 0:
            s = s + Q_value[state][act] / (Q_value[state][best_policy[state]])
        else:
            s = s + (Q_value[state][act] - Q_value[state][worst_policy[state]]) \
                / (Q_value[state][best_policy[state]] - Q_value[state][worst_policy[state]])
        if i == row - 2:
            if_end = 1
        elif list_sick[i].id != list_sick[i + 1].id:
            if_end = 1
        if if_end == 1:
            step = list_sick[i].step
            if min_r <= r / step <= max_r:
                r_line.append(r / step)
                death_line.append(death)
            if death == 0:
                s_line_live.append(s / step)
            else:
                s_line_death.append(s / step)
    print(np.mean(s_line_live), np.mean(s_line_death))
    return r_line, death_line, state_line


def value_class(section=50):
    max_r = 2
    min_r = -2
    class_num = int((max_r - min_r) / section)
    r_patient_num = np.zeros(class_num, float)
    death_num = np.zeros(class_num)
    live_num = np.zeros(class_num)
    r_live = []
    r_death = []
    for i in range(len(r_patient)):
        if r_patient[i] >= 0:
            index = int(r_patient[i]/ section) + abs(int(min_r / section))
            r_patient_num[index] += 1
            if death_patient[i] == 0:
                live_num[index] += 1
                r_live.append(r_patient[i])
            else:
                death_num[index] += 1
                r_death.append(r_patient[i])
        else:
            index = int(r_patient[i] / section) + abs(int(min_r / section))
            r_patient_num[index] += 1
            if death_patient[i] == 0:
                live_num[index] += 1
                r_live.append(r_patient[i])
            else:
                death_num[index] += 1
                r_death.append(r_patient[i])
    live_radius = np.zeros(class_num)
    print(live_num)
    print(death_num)
    for i in range(class_num):
        if live_num[i] + death_num[i] == 0:
            continue
        else:
            live_radius[i] = live_num[i] / (live_num[i] + death_num[i])
    print(live_radius)
    print(np.mean(r_live), np.mean(r_death), 'x')
    # print(r_patient_num, live_num, death_num, live_num / (live_num + death_num), np.mean(r_live), np.mean(r_death))
    plot_picture(class_num, live_radius)
    return r_patient_num, live_num, death_num, live_radius


section = 1
min_r = -2
max_r = 2
r_patient, death_patient, state_patient = value_cau(min_r, max_r)
r_patient_class_num, live_patient_num, death_patient_num, radius= value_class(section)
