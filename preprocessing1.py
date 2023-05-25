# coding=utf-8
import numpy as np
import xlrd2

path = 'RL-for-sepsis\Table_sofa.xlsx'


class sick:
    def __init__(self):
        self.id = ''
        self.step = ''
        self.score = ''


lissick = []
table = xlrd2.open_workbook(path).sheets()[0]
row = table.nrows
col = table.ncols
for x in range(row):
    if (x == 0):
        continue
    rows = np.array(table.row_values(x))
    lissick.append(sick())
    lissick[x - 1].id = rows[0]
    lissick[x - 1].step = int(rows[1])
    lissick[x - 1].score = int(rows[2])
for x in range(row - 1):
    n = lissick[x].step
    if n <= 0 and n % 4 == 0:
        if lissick[x-3].id == lissick[x].id:
            lissick[x].score = (lissick[x].score + lissick[x-1].score + lissick[x-2].score + lissick[x-3].score) / 4
        elif lissick[x-2].id == lissick[x].id:
            lissick[x].score = (lissick[x].score + lissick[x - 1].score + lissick[x - 2].score) / 3
        elif lissick[x-1].id == lissick[x].id:
            lissick[x].score = (lissick[x].score + lissick[x - 1].score) / 2
    if (x < row - 2) and (n % 4 != 0) :
        if lissick[x].id != lissick[x+1].id:
            if n % 4 == 2:
                lissick[x].score = (lissick[x].score + lissick[x-1].score) / 2
            elif n % 4 == 3:
                lissick[x].score = (lissick[x].score + lissick[x - 1].score + lissick[x-2].score) / 3
    if n % 4 == 0 and n > 0:
        lissick[x].score = (lissick[x].score + lissick[x-1].score + lissick[x-2].score + lissick[x-3].score) / 4


output = open('Table_1.xls', 'w', encoding='gbk')
output.write('id\tn\tSOFA\n')
for i in range(row - 1):
    n = lissick[i].step
    if n % 4 == 0 and n >= 0:
        output.write(str(int(round(lissick[i].id))))
        output.write('\t')
        output.write(str(lissick[i].step/4))
        output.write('\t')
        output.write(str(lissick[i].score))
        output.write('\n')
output.close()