# coding=utf-8
import numpy as np


def cau_best_policy(Q_value):
    best_policy = []
    for i in range(5):
        best_policy.append(Q_value[i].index(max(Q_value[i])))
    return best_policy


def cau_worst_policy(Q_value):
    best_policy = []
    for i in range(5):
        best_policy.append(Q_value[i].index(min(Q_value[i])))
    return best_policy


def relative_gain_cau(row, list_sick, Q_value):
    s_line_live = []
    s_line_death = []
    best_policy = cau_best_policy(Q_value)
    worst_policy = cau_worst_policy(Q_value)
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
            s = 0
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
            if death == 0:
                s_line_live.append(s / step)
            else:
                s_line_death.append(s / step)
    return np.mean(s_line_live), np.mean(s_line_death)