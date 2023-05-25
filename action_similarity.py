# coding=utf-8
import numpy as np


def cau_best_policy(Q_value):
    best_policy = []
    for i in range(5):
        best_policy.append(Q_value[i].index(max(Q_value[i])))
    return best_policy


def act_similarity_cau(row, list_sick, Q_value):
    true_act = 0
    all_act = 0
    radius_line = []
    live_rate = []
    death_rate = []
    best_policy = cau_best_policy(Q_value)
    for i in range(row - 1):
        if_end = 0
        act = list_sick[i].act
        state = list_sick[i].state
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
            if death == 0:
                live_rate.append(radius)
            else:
                death_rate.append(radius)
    return np.mean(live_rate), np.mean(death_rate)
