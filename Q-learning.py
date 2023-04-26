# coding=utf-8
import numpy as np
import xlrd2

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


def max_value(Q, state):
    max = -100
    for i in range(5):
        if Q[state][i] > max:
            max = Q[state][i]
    return max


def max_act(Q, state):
    max = -100
    for i in range(5):
        if Q[state][i] > max:
            max = Q[state][i]
    return i


def if_max(state, act):
    max = -100
    for i in range(5):
        if Q_value[state][i] > max:
            max = Q_value[state][i]
            max_act = i
    if max_act == act:
        return 1
    else:
        return 0


def state_class(state):
    if state < 2:
        return 0
    elif state < 7:
        return 0
    elif state < 12:
        return 1
    elif state < 19:
        return 1
    else:
        return 4


def eval_q_sofa():
    i = 0
    start = 0
    P_sofa = 0
    Q_sofa = 0
    F_sofa = 0
    n = 0
    while i < trajectory_num:
        # print('i =', i)
        flag = 0
        if i != trajectory_num - 1:
            id = list_sick[i].id
            next_id = list_sick[i + 1].id
            if id == next_id:
                i = i + 1
                flag = 1
        if flag == 1:
            continue
        n = n + 1
        step = list_sick[i].step
        state = list_sick[start].state
        sofa = list_sick[start].sofa
        start_sofa = sofa
        j = 0
        while j < step:
            if sofa > 24:
                sofa = 24
                break
            if sofa < 0:
                sofa = 0
                break
            reward = max_value(Q_value, state)
            sofa = sofa - reward
            state = state_class(sofa)
            j = j + 1
        start = i + 1
        F_sofa = start_sofa + F_sofa
        P_sofa = list_sick[i].sofa + P_sofa
        Q_sofa = sofa + Q_sofa
        i = i + 1
    return F_sofa, P_sofa, Q_sofa, n


def eval_p_sofa():
    i = 0
    start = 0
    n = 1
    Q_sofa = 0
    while i < trajectory_num:
        # print('i =', i)
        flag = 0
        if i != trajectory_num - 1:
            id = list_sick[i].id
            next_id = list_sick[i + 1].id
            if id == next_id:
                i = i + 1
                flag = 1
        if flag == 1:
            continue
        n = n + 1
        step = list_sick[i].step
        state = list_sick[start].state
        act = list_sick[start].act
        sofa = list_sick[start].sofa
        j = 0
        while j < step:
            reward = P_value[state][act]
            sofa = sofa - reward
            if sofa > 24:
                sofa = 24
                break
            if sofa < 0:
                sofa = 0
                break
            # print(start + j, sofa, list_sick[start + j].sofa)
            state = list_sick[start + j].state
            act = list_sick[start + j].act
            j = j + 1
        start = i + 1
        Q_sofa = sofa + Q_sofa
        i = i + 1
    return Q_sofa


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

Q_value = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]  # Q[state][act]

Q_base_value = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

P_value = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

trajectory_num = row - 1
gamma = 0.7
alpha = 0.2
num = 0

for i in range(trajectory_num - 1):
    # for j in range(5):
    #     Q_value[j][0] = 0
    # for k in range(6):
    #     for j in range(5):
    #         Q_value[k][j] = Q_value[k][j] - Q_value[k][0]
    id = list_sick[i].id
    step = list_sick[i].step
    state = list_sick[i].state
    act = list_sick[i].act
    reward = list_sick[i].reward
    death = list_sick[i].death
    if state == 1:
        state = 0
    elif state >= 2:
        state = 1
    if death == 1:
        state = 2
    if id != list_sick[i + 1].id:
        continue
    if state == 2:
        print(id, step)
    next_state = list_sick[i + 1].state
    if next_state == 1:
        next_state = 0
    elif next_state >= 2:
        next_state = 1
    next_act = list_sick[i + 1].act
    next_death = list_sick[i + 1].death
    if next_death == 1:
        next_state = 2
        reward = reward - 15
    Q_value[state][act] = Q_value[state][act] + alpha * (
            reward + gamma * max_value(Q_value, next_state) - Q_value[state][act])
    P_value[state][act] = P_value[state][act] + alpha * (
            reward + gamma * P_value[next_state][next_act] - P_value[state][act])
# f_sofa, p_sofa, q_sofa, sick_num = eval_q_sofa()
# p_sofa_1 = eval_p_sofa()
# print(f_sofa / sick_num, p_sofa / sick_num, p_sofa_1 / sick_num, q_sofa / sick_num, sick_num)

print('Q-learning')
for i in range(3):
    print(Q_value[i])
print('Sarsa')
for i in range(3):
    print(P_value[i])
# for i in range(6):
#     for j in range(5):
#         Q_base_value[i][j] = Q_value[i][j] - Q_value[i][0]
# print('修正后Q-learning')
# for i in range(6):
#     print(Q_base_value[i])
