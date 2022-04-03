from cProfile import label
from random import random
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from math import gamma


l = float(input("Интенсивность потока машин: "))
t = float(input("Время: "))
from math import exp, factorial

experiments = 1000
results = []
distr = {}

# рандомим цифру
# стартуем форик в котором поочередно по формуле с е считаем вероятности. Если рандомная циферка больше, то идем на следующую итерацию

for i in range(experiments):
    p = exp(-l * t)
    p_ = 0
    u = random()
    k = 0
    while u > p:
        p_ = t ** k * l ** k * exp(-l * t) / factorial(k)
        p += p_
        k += 1
    results.append(k)

summary = dict(Counter(results))

print(f"After {experiments} experiments:")
for k in sorted(summary.items()):
    print(
        str(k[0]) + " cars,", str(k[1]) + " times,", str(k[1] * 100 / experiments) + "%"
    )


# 2 ЧАСТЬ
# ряд распределения
unique_results = list(set(results))
for j in range(len(unique_results)):
    new_p = l ** j * t ** j * exp(-l * t) / factorial(j)
    distr[j] = new_p

x = list(distr.keys())
p = list(distr.values())

# Выборочная функция распределения
F_v = []
F_v.append(0)
n = len(distr)
for k in sorted(summary.items()):
    F_v.append(k[1] / experiments + F_v[-1])


F = []
F.append(0)
for i in x:
    F.append(l ** i * t ** i * exp(-l * t) / factorial(i) + F[-1])

# Выборочное среднее
x_ = 0
for i in x:
    x_ += i
x_ = x_ / len(x)


# Математическое ожидание
E = 0
for i in range(len(x)):
    E += x[i] * p[i]


# |E-x_|
E_x_ = abs(E - x_)

# Дисперсия
E2 = 0
for i in range(len(x)):
    E2 += x[i] * x[i] * p[i]
D = E2 - E * E

# Выборочная дисперсия
S2 = 0
for i in range(len(x)):
    S2 += (x[i] - x_) ** 2
S2 = S2 / len(x)


# D-S2
D_S2 = abs(D - S2)

# R
R = x[-1] - x[0]

# Me
if n % 2 == 0:
    Me = (x[int(n / 2)] + x[int(n / 2 + 1)]) / 2
else:
    Me = x[int((n - 1) / 2 + 1)]

# Мера расхождения
DD = 0
DD_ = []
for i in range(len(F)):
    DD_.append(abs(F[i] - F_v[i]))
DD = max(DD_)

F_ = []
for k in range(len(summary)):
    F_.append(abs(sorted(summary.items())[k][1] / experiments - p[k]))

nj = []
for k in range(len(summary)):
    nj.append(sorted(summary.items())[k][1] / experiments)


F_max = max(F_)


names = ["E", "x_", "|E-x_|", "D", "S2", "|D-S2|", "Me", "R"]
chars = [E, x_, E_x_, D, S2, D_S2, Me, R]
npchars = np.array(chars)
npnames = np.array(names)
table = pd.DataFrame(npchars, npnames).T
print(table)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Мера расхождения: ", DD)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
table2 = pd.DataFrame()
table2.index = x
table2.index.names = ["yj"]
table2["P({n = yj})"] = p
table2["nj/n"] = nj
print(table2.T)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Максимальное отклонение:")
print(F_max)


xplot = x.insert(0, -1)
plt.plot(x, F, label="Теоретическая фр")
plt.plot(x, F_v, label="Выборочная фр")
plt.legend(loc="best")
plt.show()


# 3 ЧАСТЬ

print("Введите количество интервалов k")
k_ = int(input())
print("Введите границы интервалов друг за другом")
intervals = []
inputed = input()
inputed = str.split(inputed)
for i in range(len(inputed)):
    if (i + 1) % 2 == 0:
        intervals.append((inputed[i - 1], inputed[i]))
print(intervals)

x = x[1:]

# print(intervals[2])
# print(list(range(3)))

deltas = []


def countQj(j):
    # print(j)
    interval = list(intervals[j])
    # print(interval)
    if interval[0] == "-inf":
        interval[0] = x[0]
    if interval[1] == "inf":
        interval[1] = x[-1] + 1

    ai = x[int(interval[0]) : int(interval[1])]
    deltas.append(len(ai))
    qj = [distr[x] for x in ai]
    # print(qj)
    return sum(qj)


qi = []
for i in range(k_):
    qi.append(countQj(i))
print(qi)
print(deltas)

print("Введите уровень значимости a:")
a = float(input())

R0 = 0
for i in range(k_):
    R0 += ((deltas[i] - len(x) * qi[i]) ** 2) / (len(x) * qi[i])
print(R0)

df = k_ - 1

F_hi = 0
for i in range(k_):
    F_hi += (
        ((R0 * ((i) / k_)) ** (df / 2 - 1)) * exp(-R0 * i / (k_ * 2))
        + ((R0 * ((i + 1) / k_)) ** (df / 2 - 1)) * exp(-R0 * (i + 1) / (k_ * 2))
    ) * (R0 / (2 * k_))

F_hi = F_hi * (1 / ((2 ** df) * gamma(df / 2)))

F_hi = 1 - F_hi

print(F_hi)
if F_hi < a:
    print("FALSE")
else:
    print("TRUE")
