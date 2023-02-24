import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable as PT


Matrix = [
    [[1, 10], [2, 6], [3, 6], [2, 4]],
    [[4, 10], [7, 5], [6, 4], [4, 8]],
    [[8, 7], [9, 7], [3, 2], [5, 3]],
    [[1, 3], [4, 8], [2, 6], [2, 5]],
    [[5, 3], [4, 10], [2, 6], [3, 4]],
    [[2, 9], [3, 6], [3, 12], [4, 8]],
    [[3, 6], [5, 8], [5, 7], [3, 11]],
    [[7, 12], [4, 9], [7, 7], [4, 8]]
]

# First task

# vec = [[x,y], ... ]
def get_indxs(vec, pareto_set):
    res = []
    for i in range(len(pareto_set)):
        for j in range(len(vec)):
            if pareto_set[i] == vec[j]:
                res.append(j)
                break
    return res
    
# min by rows
# res -> [[x, y], ... ]
def vec_minimax(Matrix):
    res = []
    for row in Matrix:
        min_1 = 999
        min_2 = 999
        for vec in row:
            if vec[0] < min_1:
                min_1 = vec[0]
            if vec[1] < min_2:
                min_2 = vec[1]
        res.append([min_1, min_2])
    return res

def get_one_arg(matrix_x_y, param):
    res = []
    for points in matrix_x_y:
        res.append(points[param])
    return res

# vec = [[x,y], ... ]
def print_points(vec, ax : plt.Axes):
    x = get_one_arg(vec, 0)
    y = get_one_arg(vec, 1)
    ax.grid(True)
    ax.scatter(x, y)

# if minimax = False, it'll return maximin
# result -> [[x,y], ... ]
def pareto(pts, minimax = True):
    result = []
    for pt_1 in pts:
        better_pts = False
        for pt_2 in pts:
            if minimax:
                if pt_1 != pt_2 and pt_2[0] >= pt_1[0] and pt_2[1] >= pt_1[1]:
                    better_pts = True
                    break 
            else:
                if pt_1 != pt_2 and pt_2[0] <= pt_1[0] and pt_2[1] <= pt_1[1]:
                    better_pts = True
                    break 
        if not better_pts:
            result.append(pt_1)
    return result

minimax_pts = vec_minimax(Matrix)
sol_1 = pareto(minimax_pts)

summary_0_1 = get_indxs(minimax_pts, sol_1)

# print first task
fig, ax = plt.subplots()
ax.set_title('По принципу векторного минимакса')
print_points(minimax_pts, ax)
print_points(sol_1, ax, )
x = get_one_arg(minimax_pts, 0)
y = get_one_arg(minimax_pts, 1)
for i, txt in enumerate(range(1, len(minimax_pts) + 1)):
    ax.annotate(txt, (x[i], y[i]))

# Second task

# get matrix of the best points
# res -> [[x, y], ... ]
def vec_minimax_regret(Matrix):
    res = []
    m_range = range(len(Matrix))
    r_range = range(len(Matrix[0]))
    for j in r_range:
        max_1 = 0
        max_2 = 0
        for i in m_range:
            pt = Matrix[i][j]
            if pt[0] > max_1:
                max_1 = pt[0]
            if pt[1] > max_2:
                max_2 = pt[1]
        res.append([max_1, max_2])
    return res

# res -> same as a Matrix
def get_regret_matrix(b_pts_matrix, Matrix):
    res = np.zeros_like(Matrix).tolist()
    m_range = range(len(Matrix))
    r_range = range(len(Matrix[0]))
    for j in r_range:
        for i in m_range:
            pt_m = Matrix[i][j]
            pt_b = b_pts_matrix[j]
            res[i][j] = [pt_b[0] - pt_m[0], pt_b[1] - pt_m[1]]
    return res

# res -> [[x,y], ... ]
def vec_maximin(U_Matrix):
    res = []
    for row in U_Matrix:
        max_1 = 0
        max_2 = 0
        for vec in row:
            if vec[0] > max_1:
                max_1 = vec[0]
            if vec[1] > max_2:
                max_2 = vec[1]
        res.append([max_1, max_2])
    return res

best_pts = vec_minimax_regret(Matrix)
u_matrix = get_regret_matrix(best_pts, Matrix) 
maximin_pts = vec_maximin(u_matrix)
sol_2 = pareto(maximin_pts, False)
summary_0_2 = get_indxs(maximin_pts, sol_2)

# print second task
fig, ax = plt.subplots()
ax.set_title('По принципу векторного минимаксного сожаления')
print_points(maximin_pts, ax)
print_points(sol_2, ax)
x = get_one_arg(maximin_pts, 0)
y = get_one_arg(maximin_pts, 1)
for i, txt in enumerate(range(1, len(maximin_pts) + 1)):
    ax.annotate(txt, (x[i], y[i]))

plt.show()

# Third task

def get_matrix_by_ax(Matrix, axis):
    res = []
    for row in Matrix:
        row_vec = []
        for vec in row:
            row_vec.append(vec[axis])
        res.append(row_vec)
    return res

def get_min_pts_r(n_Matrix):
    res = []
    for row in n_Matrix:
        min_pt = 999
        for pt in row:
            if pt < min_pt:
                min_pt = pt
        res.append(min_pt)
    return res

def get_max_pts_r(n_Matrix):
    res = []
    for row in n_Matrix:
        max_pt = 0 
        for pt in row:
            if pt > max_pt:
                max_pt = pt
        res.append(max_pt)
    return res

def get_min_indxs(vec):
    x_min = min(vec)
    res = []
    for i in range(len(vec)):
        if vec[i] == x_min:
            res.append(i)
    return res

def get_max_indxs(vec):
    x_max = max(vec)
    res = []
    for i in range(len(vec)):
        if vec[i] == x_max:
            res.append(i)
    return res

def vald(n_Matrix):
    res = get_min_pts_r(n_Matrix)
    return get_max_indxs(res) 

def savidzh(n_Matrix):
    B = []
    m_range = range(len(n_Matrix))
    r_range = range(len(n_Matrix[0]))
    for j in r_range:
        max_pt = 0
        for i in m_range:
            pt = n_Matrix[i][j]
            if pt > max_pt:
                max_pt = pt
        B.append(max_pt)
    R = []
    for i in m_range:
        max_pt = 0
        row = []
        for j in r_range:
            pt = n_Matrix[i][j]
            b = B[j]
            row.append(b - pt)
        R.append(row)
    x_ = []
    for row in R:
        x_.append(max(row))
    return get_min_indxs(x_)

def gurvits(n_Matrix):
    x_mins = get_min_pts_r(n_Matrix)
    x_maxs = get_max_pts_r(n_Matrix)
    
    C = []
    lmda = 0.6
    for i in range(len(x_maxs)):
        C.append(lmda * x_mins[i] + (1 - lmda) * x_maxs[i])
    return get_min_indxs(C)

def laplas(n_Matrix):
    p_j = 1 / len(n_Matrix[0])
    D = []
    for row in n_Matrix:
        D.append(sum(row) * p_j)
    return get_max_indxs(D)

n_Matrix = get_matrix_by_ax(Matrix, 0)
res_v_1 = [x+1 for x in vald(n_Matrix)]
res_s_1 = [x+1 for x in savidzh(n_Matrix)]
res_g_1 = [x+1 for x in gurvits(n_Matrix)]
res_l_1 = [x+1 for x in laplas(n_Matrix)]

summary_1 = [res_v_1, res_s_1, res_g_1, res_l_1]

#print('По Вальду: ' + str(res_v_1) + \
#        ' По Сэвиджу: ' + str(res_s_1) + \
#        ' По Гурвицу: ' + str(res_g_1) + \
#        ' По Лапласу: ' + str(res_l_1))


# Fourth task

nn_Matrix = get_matrix_by_ax(Matrix, 1)
res_v_2 = [x+1 for x in vald(nn_Matrix)]
res_s_2 = [x+1 for x in savidzh(nn_Matrix)]
res_g_2 = [x+1 for x in gurvits(nn_Matrix)]
res_l_2 = [x+1 for x in laplas(nn_Matrix)]

summary_2 = [res_v_2, res_s_2, res_g_2, res_l_2]

#print('По Вальду: ' + str(res_v_2) + \
#        ' По Сэвиджу: ' + str(res_s_2) + \
#        ' По Гурвицу: ' + str(res_g_2) + \
#        ' По Лапласу: ' + str(res_l_2))

# summary

stats = np.zeros((10, len(Matrix))).tolist()
for s0_1 in summary_0_1:
    stats[0][s0_1] = 1
for s0_2 in summary_0_2:
    stats[1][s0_2] = 1
i = 2
for s1 in summary_1:
    s = stats[i]
    for res in s1:
        s[res - 1] = 1
    i += 1
i = 6
for s2 in summary_2:
    s = stats[i]
    for res in s2:
        s[res - 1] = 1
    i += 1
s = stats
sum_stats = [sum([r[0] for r in s]), sum([r[1] for r in s]), \
             sum([r[2] for r in s]), sum([r[3] for r in s]), \
             sum([r[4] for r in s]), sum([r[5] for r in s]), \
             sum([r[6] for r in s]), sum([r[7] for r in s])]
ss = sum_stats
pt = PT()
pt.field_names = [" ", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
pt.add_row(["Минимакс",       s[0][0], s[0][1], s[0][2], s[0][3], s[0][4], s[0][5], s[0][6], s[0][7]])
pt.add_row(["Минимакс. сож.", s[1][0], s[1][1], s[1][2], s[1][3], s[1][4], s[1][5], s[1][6], s[1][7]])
pt.add_row(["Вальд 1",        s[2][0], s[2][1], s[2][2], s[2][3], s[2][4], s[2][5], s[2][6], s[2][7]])
pt.add_row(["Сэвидж 1",       s[3][0], s[3][1], s[3][2], s[3][3], s[3][4], s[3][5], s[3][6], s[3][7]])
pt.add_row(["Гурвиц 1",       s[4][0], s[4][1], s[4][2], s[4][3], s[4][4], s[4][5], s[4][6], s[4][7]])
pt.add_row(["Лаплас 1",       s[5][0], s[5][1], s[5][2], s[5][3], s[5][4], s[5][5], s[5][6], s[5][7]])
pt.add_row(["Вальд 2",        s[6][0], s[6][1], s[6][2], s[6][3], s[6][4], s[6][5], s[6][6], s[6][7]])
pt.add_row(["Сэвидж 2",       s[7][0], s[7][1], s[7][2], s[7][3], s[7][4], s[7][5], s[7][6], s[7][7]])
pt.add_row(["Гурвиц 2",       s[8][0], s[8][1], s[8][2], s[8][3], s[8][4], s[8][5], s[8][6], s[8][7]])
pt.add_row(["Лаплас 2",       s[9][0], s[9][1], s[9][2], s[9][3], s[9][4], s[9][5], s[9][6], s[9][7]])
pt.add_row(["Summary",        ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], ss[6], ss[7]])

print(pt)
print()
print('Best choice is ' + str([x+1 for x in get_max_indxs(ss)]))
