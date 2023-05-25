# coding=utf-8
import math


def if_max(Q_value, state, act):
    value = Q_value[state][act]
    for i in range(5):
        if Q_value[state][i] > value:
            return 0
    return 1


def prob_cau(number, prob):
    sum = [0, 0, 0, 0, 0]
    for i in range(5):
        for j in range(5):
            sum[i] = sum[i] + number[i][j]
    for i in range(5):
        for j in range(5):
            if sum[i] == 0:
                prob[i][j] = 0
                continue
            prob[i][j] = number[i][j] / sum[i]
    return prob


def rou_cau(trajectory_num, list_sick, number_s_a, prob):
    for i in range(trajectory_num):
        state = list_sick[i].state
        act = list_sick[i].act
        if state != 5:
            number_s_a[state][act] = number_s_a[state][act] + 1
    prob = prob_cau(number_s_a, prob=prob)
    return prob


def TD_offline(Q_value, prob, trajectory_num, gamma, alpha, n, list_sick):
    for i in range(trajectory_num):
        if_start = 0
        if i == 0:
            if_start = 1
        elif list_sick[i].id != list_sick[i - 1].id:
            if_start = 1
        if if_start == 0:
            continue
        T = 100000
        for t in range(100000):
            t = t + i
            if t < T:
                if_end = 0
                if t + 1 == trajectory_num - 1:
                    if_end = 1
                elif list_sick[t + 1].id != list_sick[t + 2].id:
                    if_end = 1
                if if_end == 1:
                    T = t + 1
            tao = t - n + 1
            if tao - i >= 0:
                G = 0
                rou = 1
                for j in range(tao + 1, min(tao + n, T) + 1):
                    G = G + math.pow(gamma, j - tao - 1) * list_sick[j].reward
                for j in range(tao + 1, min(tao + n - 1, T - 1) + 1):
                    rou = rou * if_max(Q_value, list_sick[j].state, list_sick[j].act) / prob[list_sick[j].state][
                        list_sick[j].act]
                if tao + n < T:
                    state = list_sick[tao + n].state
                    act = list_sick[tao + n].act
                    G = G + math.pow(gamma, n) * Q_value[state][act]
                state = list_sick[tao].state
                act = list_sick[tao].act
                Q_value[state][act] = Q_value[state][act] + alpha * rou * (G - Q_value[state][act])
            if tao == T - 1:
                break
    return Q_value
