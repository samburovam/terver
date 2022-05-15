from cProfile import label
from random import random
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from math import gamma, ceil
import scipy.integrate as integrate
from math import exp, factorial

l = 2
t = 2


def run(k_, a):
    # l = float(input("Интенсивность потока машин: "))
    # t = float(input("Время: "))

    experiments = 1000
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
            str(k[0]) + " cars,",
            str(k[1]) + " times,",
            str(k[1] * 100 / experiments) + "%",
        )
    x_v = np.array(x_v)
    p_v = np.array(p_v)
    print(p_v)

    def Poisson(x):
        return l ** x * t ** x * exp(-l * t) / factorial(x)

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

    """
    plt.step(x_v, F[1:], label="Теоретическая фр")
    plt.step(x_v, F_v[1:], label="Выборочная фр")
    plt.legend(loc="best")
    plt.show()
    """

    # 3 ЧАСТЬ

    """
    print("Введите границы интервалов друг за другом")
    intervals = []
    inputed = input()
    inputed = str.split(inputed)
    for i in range(len(inputed)):
        if (i + 1) % 2 == 0:
            intervals.append((inputed[i - 1], inputed[i]))
    print(intervals)



    """

    def F(k):
        sum = 0
        for i in range(k + 1):
            sum += Poisson(i)
        return sum

    proms = [x / k_ for x in range(k_ + 1)]
    print(proms)

    ints = []

    for i in proms:
        for ai in x:
            if F(ai) < i and F(ai + 1) > i:
                ints.append(((F(ai + 1) - F(ai)) / 2) + F(ai))

    print(ints)

    promies = []

    ints.insert(0, 0)
    ints.insert(len(ints), 1)
    print(ints)

    newints = []

    for i in range(len(ints)):
        if i == 0:
            newints.append((0, ints[1]))
        elif i != 0 and i != len(ints) - 1:
            newints.append((ints[i], ints[i + 1]))

    print(newints)

    values = []
    qis = []
    for new in newints:
        vals = []
        qii = 0
        for i in range(len(x)):
            if F(x[i]) >= new[0] and F(x[i]) < new[1]:
                vals.append(x[i])
                qii += p[i]
        values.append(vals)
        qis.append(qii)
    print(values)
    print(qis)

    nummies = []
    for j in values:
        num = 0
        for i in results:
            if i in j:
                num += 1
        nummies.append(num)
    print(nummies)

    eta = nummies
    qi = qis

    """
    # print(intervals[2])
    # print(list(range(3)))
    eta = []
    for m in intervals:
        m = list(m)
        cou = 0
        if m[0] == "-inf":
            m[0] = x[0]
        if m[1] == "inf":
            m[1] = x[-1] + 1

        for i in results:
            if (i >= float(m[0])) and (i < float(m[1])):
                cou += 1
        eta.append(cou)


    def countQj(j):
        # print(j)
        interval = list(intervals[j])
        # print(interval)
        if interval[0] == "-inf":
            interval[0] = x[0]
        if interval[1] == "inf":
            interval[1] = x[-1] + 1

        ai = x[int(ceil(float(interval[0]))) : int(ceil(float(interval[1])))]
        print(ai)
        qj = []
        for e in ai:
            qj.append(Poisson(e))

        return sum(qj)


    qi = []
    for i in range(k_):
        qi.append(countQj(i))

    """
    print("Теоретические вероятности qi:")
    print(qi)

    print(eta)
    R0 = 0
    for i in range(k_):
        R0 += ((eta[i] - experiments * qi[i]) ** 2) / (experiments * qi[i])

    print("R0 равно:")
    print(R0)

    df = k_ - 1

    def fx(x):
        if x <= 0:
            return 0
        else:
            y = 2 ** (-df / 2)
            y = y * gamma(df / 2) ** (-1)
            y = y * x ** (df / 2 - 1)
            y = y * exp(-x / 2)
            return y

    F_hi = integrate.quad(fx, 0, R0)

    F_hi = 1.0 - float(F_hi[0])

    print("F_(R0) равно:")
    print(F_hi)
    if F_hi < a:
        return 0
    else:
        return 1


yes = 0
no = 0
for i in range(100):
    h = run(4, 0.6)
    if h == 0:
        no += 1
    else:
        yes += 1

print(yes, "accepted")
print(no)
