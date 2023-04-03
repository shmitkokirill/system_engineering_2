from prettytable import PrettyTable

start_index = 1
matrix = [[8, 7, 2, 1], [1, 2, 5, 5]]
steps_count = 3000

rowb = matrix[start_index - 1].copy()
rowa = [0, 0]
alfa = 0
beta = 0
v = 0
i = start_index
ps = [0 for t in range(len(matrix))]
qs = [0 for t in range(len(matrix[0]))]
table = PrettyTable()
table.field_names = (
    ["k", "i"]
    + ["B" + str(i + 1) for i in range(len(matrix[0]))]
    + ["α", "j"]
    + ["A" + str(i + 1) for i in range(len(matrix))]
    + ["β", "v"]
)


def sum_col(a, b):
    return [x + y for x, y in zip(a, b)]


for k in range(1, steps_count + 1):
    min_rb = min(rowb)
    alfa = min_rb / k
    j = rowb.index(min_rb) + 1
    qs[j - 1] += 1
    rowa = sum_col(rowa, [matrix[k][j - 1] for k in range(len(matrix))])
    max_ra = max(rowa)
    beta = max_ra / k
    v = (alfa + beta) / 2

    table.add_row(
        [str(k), str(i)] + rowb + [str(alfa), str(j)] + rowa + [str(beta), str(v)]
    )
    i = rowa.index(max_ra) + 1
    ps[i - 1] += 1
    rowb = sum_col(rowb, matrix[i - 1])

print(table)
print("p =", [ps[i] / k for i in range(len(ps))])
print("q =", [qs[i] / k for i in range(len(qs))])
print("v =", v)
