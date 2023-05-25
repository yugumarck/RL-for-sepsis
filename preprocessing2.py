# coding=utf-8
import numpy as np
import xlrd2

path_state = 'RL-for-sepsis\Table_1.xlsx'
path_act = 'RL-for-sepsis\Table_heparin.xlsx'


class sick_state:
    def __init__(self):
        self.id = ''
        self.step = ''
        self.score = ''


class sick_act:
    def __init__(self):
        self.id = ''
        self.step = ''
        self.act = ''


class sick:
    def __init__(self):
        self.id = ''
        self.step = ''
        self.state = ''
        self.act = ''
        self.reward = ''
        self.death = ''
        self.sofa = ''


def act_class(act):
    if act == 0:
        return 0
    elif act <= 1.38:
        return 1
    elif act <= 1.88:
        return 2
    elif act <= 3.5:
        return 3
    else:
        return 4


def state_class(state):
    if state < 2:
        return 0
    elif state < 7:
        return 1
    elif state < 12:
        return 2
    elif state < 19:
        return 3
    elif state < 24:
        return 4
    else:
        return 5


list_state = []
list_act = []
list_sick = []

table_s = xlrd2.open_workbook(path_state).sheets()[0]
row_s = table_s.nrows
col_s = table_s.ncols

table_a = xlrd2.open_workbook(path_act).sheets()[0]
row_a = table_a.nrows
col_a = table_a.ncols

for x in range(row_s):
    if x == 0:
        continue
    rows_s = np.array(table_s.row_values(x))
    list_state.append(sick_state())
    list_state[x - 1].id = int(rows_s[0])
    list_state[x - 1].step = int(rows_s[1])
    list_state[x - 1].score = float(rows_s[2])

for x in range(row_a):
    if x == 0:
        continue
    rows_a = np.array(table_a.row_values(x))
    list_act.append(sick_act())
    list_act[x - 1].id = int(rows_a[0])
    list_act[x - 1].step = int(rows_a[1])
    if list_act[x - 1].step <= 0:
        list_act[x - 1].step = list_act[x - 1].step + 1
    list_act[x - 1].act = float(rows_a[2])

num = -1
i = 0
j = 0
while i < row_s - 1 and j < row_a - 1:
    state_id = list_state[i].id
    act_id = list_act[j].id
    if state_id == act_id:
        if list_state[i].step != list_act[j].step:
            if list_state[i].step < list_act[j].step:
                i = i + 1
            else:
                j = j + 1
            continue
        num = num + 1
        list_sick.append(sick())
        list_sick[num].id = state_id
        list_sick[num].step = list_act[j].step
        list_sick[num].state = state_class(list_state[i].score)
        list_sick[num].act = act_class(list_act[j].act)
        list_sick[num].sofa = list_state[i].score
        if i == row_s - 2:
            list_sick[num].reward = 0
        elif list_state[i].id == list_state[i + 1].id:
            list_sick[num].reward = list_state[i].score - list_state[i + 1].score
        else:
            list_sick[num].reward = 0
        if list_sick[num].state == 5:
            list_sick[num].death = 1
            list_sick[num].reward -= 15
        else:
            list_sick[num].death = 0
        i = i + 1
        j = j + 1
    if state_id != act_id:
        if list_state[i].id < list_act[j].id:
            i = i + 1
        else:
            j = j + 1

output = open('Table_data.xls', 'w', encoding='gbk')
output.write('id\tn\tstate\tact\treward\tsofa\n')
for i in range(num + 1):
    output.write(str(int(round(list_sick[i].id))))
    output.write('\t')
    output.write(str(int(round(list_sick[i].step))))
    output.write('\t')
    output.write(str(list_sick[i].state))
    output.write('\t')
    output.write(str(list_sick[i].act))
    output.write('\t')
    output.write(str(list_sick[i].reward))
    output.write('\t')
    output.write(str(list_sick[i].sofa))
    output.write('\n')
output.close()