# coding=utf-8
def Q_TD_cau(P_value, trajectory_num, gamma, alpha, list_sick):
    for i in range(trajectory_num - 1):
        id = list_sick[i].id
        state = list_sick[i].state
        act = list_sick[i].act
        reward = list_sick[i].reward
        death = list_sick[i].death
        if death == 1:
            state = 5
        if id != list_sick[i + 1].id:
            continue
        next_state = list_sick[i + 1].state
        next_death = list_sick[i + 1].death
        if next_death == 1:
            next_state = 5
            reward = reward - 15
        P_value[state][act] = P_value[state][act] + alpha * (
                reward + gamma * max(P_value[next_state]) - P_value[state][act])
    return P_value


def sign(a, b):
    if a > b:
        return 1
    else:
        return 0


def similarity_cau(Q_value, P_value):

    flag_q = 0
    for i in range(5):
        for j in range(4):
            if sign(P_value[i][j], P_value[i][j + 1]) == sign(Q_value[i][j], Q_value[i][j + 1]):
                flag_q += 1
    return flag_q
