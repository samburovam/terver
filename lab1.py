from random import random
from collections import Counter
from matplotlib import pyplot as plt

from numpy import exp2


l = int(input("Интенсивность потока машин: "))
t = int(input("Время: "))
from math import exp, factorial

experiments = 1000
results = []

# рандомим цифру
# стартуем форик в котором поочередно по формуле с е считаем вероятности. Если рандомная циферка больше, то идем на следующую итерацию

for i in range(experiments):
    p = t * l * exp(-l * t)
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

summary_items = dict(sorted(summary.items(), key=lambda x: x[0]))
cars = list(summary_items.keys())
times = list(summary_items.values())
plt.plot(cars, times)
plt.xlabel("times")
plt.ylabel("cars")
plt.show()
