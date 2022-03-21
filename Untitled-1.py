from cProfile import label
from random import random
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


l = float(input("Интенсивность потока машин: "))
t = float(input("Время: "))
from math import exp, factorial

experiments = 10000
results = []
distr = {}
x_v = []
p_v = []

# рандомим цифру
# стартуем форик в котором поочередно по формуле с е считаем вероятности. Если рандомная циферка больше, то идем на следующую итерацию

for i in range(experiments):
    p = exp(-l * t)
    p_ = 0
    u = random()
    k = 0
    while u > p:
        k += 1
        p_ = t ** k * l ** k * exp(-l * t) / factorial(k)
        p += p_
    results.append(k)


summary = dict(Counter(results))

print(f"After {experiments} experiments:")
for k in sorted(summary.items()):
    x_v.append(k[0])
    p_v.append(k[1] / experiments)
    print(
        str(k[0]) + " cars,", str(k[1]) + " times,", str(k[1] * 100 / experiments) + "%"
    )
x_v = np.array(x_v)
p_v = np.array(p_v)
print(p_v)

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
for i in x_v:
    F.append(l ** i * t ** i * exp(-l * t) / factorial(i) + F[-1])

# Выборочное среднее
x_ = 0
for i in results:
    x_ += i
x_ = x_ / len(results)


# Математическое ожидание
E = l * t

# |E-x_|
E_x_ = abs(E - x_)

# Дисперсия
D = l * t

# Выборочная дисперсия
S2 = 0
for i in range(len(results)):
    S2 += (results[i] - x_) ** 2
S2 = S2 / len(results)


# D-S2
D_S2 = abs(D - S2)

# R
R = x_v[-1] - x_v[0]

# Me
n = len(results)
if n % 2 == 0:
    Me = (results[int(n / 2)] + results[int(n / 2 + 1)]) / 2
else:
    Me = results[int((n - 1) / 2 + 1)]

# Мера расхождения
DD = 0
DD_ = []
for i in range(len(F)):
    DD_.append(abs(F[i] - F_v[i]))
DD = max(DD_)

# Максимальное отклонение
maxF = 0
imaxF = 0
m = len(p_v)
for i in range(m):
    if abs(p_v[i] - p[i]) > maxF:
        maxF = abs(p_v[i] - p[i])
        imaxF = p_v[i]


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
table2.index = x_v
table2.index.names = ["yj"]
table2["P({n = yj})"] = np.array(p)
table2["nj/n"] = p_v / len(p_v)
print(table2.T)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Максимальное отклонение:")
print(maxF)
print("В точке:")
print(imaxF)

plt.step(x_v, F[1:], label="Теоретическая фр")
plt.step(x_v, F_v[1:], label="Выборочная фр")
plt.legend(loc="best")
plt.show()