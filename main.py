# coding=utf-8
import numpy as np
import xlrd2
from TD_lambda import TD_offline
from TD_lambda import rou_cau
from action_similarity import act_similarity_cau
from relative_gain import relative_gain_cau
from similarity import Q_TD_cau
from similarity import similarity_cau


if __name__ == '__main__':

    path = 'F:\Reinforcement Learning\sepsis\data\lab_data.xlsx'

    Q_value = []
    P_value = []
    prob = []
    number_s_a = []

    for i in range(6):
        Q_value.append([0] * 5)
        P_value.append([0] * 5)

    for i in range(5):
        prob.append([0] * 5)
        number_s_a.append([0] * 5)

    class sick:
        def __init__(self):
            self.id = ''
            self.step = ''
            self.state = ''
            self.act = ''
            self.reward = ''
            self.death = ''
            self.sofa = ''


    list_sick = []
    table = xlrd2.open_workbook(path).sheets()[0]
    row = table.nrows
    col = table.ncols

    for x in range(row):
        if x == 0:
            continue
        rows = np.array(table.row_values(x))
        list_sick.append(sick())
        list_sick[x - 1].id = rows[0]
        list_sick[x - 1].step = int(float(rows[1]))
        list_sick[x - 1].state = int(float(rows[2]))
        list_sick[x - 1].act = int(float(rows[3]))
        list_sick[x - 1].reward = float(rows[4])
        list_sick[x - 1].death = int(float(rows[5]))
        list_sick[x - 1].sofa = float(rows[6])

    prob = rou_cau(trajectory_num=row - 1, list_sick=list_sick, number_s_a=number_s_a, prob=prob)
    P_value = Q_TD_cau(P_value=P_value, trajectory_num=row - 1,
                         gamma=0.7, alpha=0.2, list_sick=list_sick)
    Q_value = TD_offline(Q_value=Q_value, prob=prob, trajectory_num=row - 1,
                         gamma=0.7, alpha=0.2, n=1, list_sick=list_sick)
    # change n to use different algorithm
    similarity = similarity_cau(Q_value, P_value)
    good_sim_a, bad_sim_a = act_similarity_cau(row, list_sick, Q_value)
    good_sim_r, bad_sim_r = relative_gain_cau(row, list_sick, Q_value)
