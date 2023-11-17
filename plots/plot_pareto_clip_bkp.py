import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from scipy.interpolate import BPoly

# Data
methods = ["MethodA", "MethodB", "MethodC"]
top10_20 = [12.8	, 20.318	, 30]
top30 = [12.6	, 27.3	, 38]
top40 = [11.87	, 26.2	, 39]
compute_32M = [12.8, 12.6, 11.87]
compute_128M = [20.318, 27.3, 26.2]
compute_640M = [30, 38, 39]

# Best points (highest value is best for each scale)
best_0 = max([top10_20[0], top30[0], top40[0]])
best_1 = max([top10_20[1], top30[1], top40[1]])
best_2 = max([top10_20[2], top30[2], top40[2]])

scales = [32, 128, 640]

widths = np.array([1, 128/32, 640/32])*4

# Define a list of markers
markers = ['*', 's', '^']  
# colors = ['skyblue', 'lightcoral', 'lightgreen']
colors = ['red', 'green', 'blue']
softened_colors = ['lightcoral', 'lightgreen', 'skyblue']

#32
dummy_scale_32 = [32-widths[0], 32, 32+widths[0]]
spline = make_interp_spline(dummy_scale_32, compute_32M, k=2)  # k is the degree of the spline
smooth_dummy_scale = np.linspace(min(dummy_scale_32), max(dummy_scale_32), 500)
smooth_compute_32M = spline(smooth_dummy_scale)
plt.plot(smooth_dummy_scale, smooth_compute_32M, color='black', linewidth=1, linestyle='dashed')

for i, (x, y, m) in enumerate(zip(dummy_scale_32, compute_32M, markers)):
    plt.plot(x, y, marker=m, color=colors[i], markersize=7)


#128
dummy_scale_128 = [128-widths[1], 128, 128+widths[1]]
spline = make_interp_spline(dummy_scale_128, compute_128M, k=2)  # k is the degree of the spline
smooth_dummy_scale = np.linspace(min(dummy_scale_128), max(dummy_scale_128), 500)
smooth_compute_128M = spline(smooth_dummy_scale)
plt.plot(smooth_dummy_scale, smooth_compute_128M, color='black', linewidth=1, linestyle='dashed')
for i, (x, y, m) in enumerate(zip(dummy_scale_128, compute_128M, markers)):
    plt.plot(x, y, marker=m, color=colors[i], markersize=7)

#640
dummy_scale_640 = [640-widths[2], 640, 640+widths[2]]
# derivatives = np.array([[0, 0],  # Derivatives before the first point
#                         [0, 0],  # Derivatives at the first point
#                         [0, 0]])
# bpoly = BPoly.from_derivatives(dummy_scale_640, derivatives)
spline = make_interp_spline(dummy_scale_640, compute_640M, k=2)  # k is the degree of the spline
smooth_dummy_scale = np.linspace(min(dummy_scale_640), max(dummy_scale_640), 10)
smooth_compute_640M = spline(smooth_dummy_scale)
#clip the curve to actual max value of best_2
# smooth_compute_640M = [min(x, best_2) for x in smooth_compute_640M]
plt.plot(smooth_dummy_scale, smooth_compute_640M, color='black', linewidth=1, linestyle='dashed')
for i, (x, y, m) in enumerate(zip(dummy_scale_640, compute_640M, markers)):
    plt.plot(x, y, marker=m, color=colors[i], markersize=7)

# plt.xscale('log', base=2)  # Set x-axis to log scale with base 2
# Scatter plot of the best points
plt.scatter(scales[0]-widths[0], best_0, marker=markers[0], color=colors[0], s=300, edgecolors='black', linewidths=1)
plt.scatter(scales[1], best_1, marker=markers[1], color=colors[1], s=200, edgecolors='black', linewidths=1)
plt.scatter(scales[2]+widths[2], best_2, marker=markers[2], color=colors[2], s=200, edgecolors='black', linewidths=1)


#add barplot corresponding to each scatter dot in the same color, have same width for each bar
midpoints = [32, 128, 640]

plt.bar(midpoints[0]-widths[0], compute_32M[0]-1.5, widths[0]*0.9, color=softened_colors[0], edgecolor='black', linewidth=0)
plt.bar(midpoints[0], compute_32M[1]-1.5, widths[0]*0.9, color=softened_colors[1], edgecolor='black', linewidth=0)
plt.bar(midpoints[0]+widths[0], compute_32M[2]-1.5, widths[0]*0.9, color=softened_colors[2], edgecolor='black', linewidth=0)

plt.bar(midpoints[1], best_1-1.5, widths[1], color=softened_colors[1], edgecolor='black', linewidth=1)
plt.bar(midpoints[2], best_2-1.5, widths[2], color=softened_colors[2], edgecolor='black', linewidth=1)


# Labels and Title
plt.xlabel('Total Training Samples', fontsize=14)
plt.ylabel('ImageNet ZeroShot Accuracy', fontsize=14)
plt.title('Data Filtering cannot be Compute Agnostic', fontsize=14)
plt.legend()

#save the plot
plt.savefig('pareto_clip.png')
