import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
def func(x, a, b):

    c=0.5
    return a*np.log(x) + b

x_values = [1, 12500, 18750, 25000, 31250]
x_values = [x*4096 for x in x_values]
y_values = [0.084, 8.48, 10.91, 12.84, 14.32]
params, _ = curve_fit(func, x_values[1:], y_values[1:])
y_values_estimated = func(np.array(x_values), *params)
print(params)
# print(func())

# Plotting
plt.figure(figsize=(25, 5))
#make a marker list
marker_list = ['o','x','^','s','*','+','D','v','p','h']
#make a color list 
color_list = ['b','g','r','c','m','y','k','w']

plt.subplot(1, 5, 1)
# for k in range(len(folder_path_list)):

plt.plot(x_values, y_values_estimated, label='no filtering')
plt.scatter(x_values, y_values, marker=marker_list[0])
# plt.plot(x_values, y_values, marker='o')
# plt.plot(x_values2, y_values2, marker='x')
plt.title('Accuracy vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_vs_samples.png")