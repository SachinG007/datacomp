import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
deltas = [0.033479999999999954, 0.02946000000000004, 0.04815999999999998, 0.059880000000000044]
y = 1 - np.cumsum(deltas)
print("error values", y)

x = [16011264, 32026624, 64016384, 127959040]
x_1 = [0, 16011264, 32026624, 64016384]
x = np.array(x)/12_800_000
x_1 = np.array(x_1)/12_800_000
indices = np.arange(1,5)

def power(x, power):
    if x==0:
        return 0
    return x**power

def func(index, a, b, half_life):
    acc = 0
    base = 0.5
    # a = 1
    b = -0.0729
    if type(index) != type(4):
        accs = []
        for j in index:
            j = int(j)
            for i in range(j):
                acc += a * (power(x[i],b) - power(x_1[i],b))
            accs.append(acc)
        return accs
    else:
        for i in range(index):
            acc += a * (power(x[i],b) - power(x_1[i],b))
    return acc

#curve fit
print("y", y)
popt, pcov = curve_fit(func, indices, y)
print("popt", popt)

#estimated values
y_est = []
for i in indices.tolist():
    y_est.append(func(i, *popt))
print("y_est", y_est)

#plot original and estimated values, scatter the original and plot the estimated
plt.scatter(x, y)
plt.plot(x, y_est, label="estimated")
plt.legend()
#save
plt.savefig("estimated.png")