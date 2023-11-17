import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from scipy.interpolate import BPoly
from matplotlib.lines import Line2D

import matplotlib as mpl

mpl.rcParams.update({
    # 'text.usetex': True,           # Use LaTeX for all text handling
    # 'font.family': 'serif',        # Use serif font instead of sans-serif
    'font.serif': 'Times',         # Specific serif font (e.g., Times)
    'axes.labelsize': 14,          # Size of axis labels
    'axes.titlesize': 16,          # Size of title
    'font.size': 14,               # Size of general text
    'legend.fontsize': 14,         # Size of legend text
    'xtick.labelsize': 14,         # Size of x-tick labels
    'ytick.labelsize': 12,         # Size of y-tick labels
    'figure.figsize': [6.4, 4.8],  # Default figure size
    'lines.linewidth': 1.5,        # Width of lines
    'lines.markersize': 6,         # Size of markers
    'axes.grid': True,             # Enable grid by default
    'grid.alpha': 0.5,             # Transparency of grid
    'grid.linestyle': '--',        # Style of grid lines
})


# Data
methods = ["32M", "128M", "640M"]
top10_20 = [12.8	, 20.318	, 30]
top30 = [12.6	, 27.3	, 38]
top40 = [11.87	, 26.2	, 39]
compute_100M = [12.8, 12.6, 11.87]
compute_200M = [20.318, 27.3, 26.2]
compute_300M = [30, 38, 39]

# Best points (highest value is best for each scale)
best_0 = max([top10_20[0], top30[0], top40[0]])
best_1 = max([top10_20[1], top30[1], top40[1]])
best_2 = max([top10_20[2], top30[2], top40[2]])

scales = [100,200,300]
markersize = np.array([10,7,7])*2
widths = np.array([1, 1, 1])*20

#make a plot connecting the best points
plt.plot([scales[0]-widths[0], scales[1], scales[2]+widths[2]], [best_0, best_1, best_2], color='darkorange', linewidth=3, linestyle='dotted')
#write pareto-frontier over this plot, tilt at 45 degree
plt.text(95, 17, 'Pareto Filtering', fontsize=16, color='black', rotation=34)

# Define a list of markers
markers = ['*', 's', '^']  
# colors = ['skyblue', 'lightcoral', 'lightgreen']
colors = ['red', 'green', 'blue']
softened_colors = ['lightcoral', 'lightgreen', 'skyblue']

#100
dummy_scale_100 = [100-widths[0], 100, 100+widths[0]]
spline = make_interp_spline(dummy_scale_100, compute_100M, k=2)  # k is the degree of the spline
smooth_dummy_scale = np.linspace(min(dummy_scale_100), max(dummy_scale_100), 500)
smooth_compute_100M = spline(smooth_dummy_scale)
plt.plot(smooth_dummy_scale, smooth_compute_100M, color='black', linewidth=1, linestyle='dashed')

for i, (x, y, m) in enumerate(zip(dummy_scale_100, compute_100M, markers)):
    plt.plot(x, y, marker=m, color=colors[i], markersize=markersize[i], markeredgecolor=softened_colors[i], zorder=10)


#200
dummy_scale_200 = [200-widths[1], 200, 200+widths[1]]
spline = make_interp_spline(dummy_scale_200, compute_200M, k=2)  # k is the degree of the spline
smooth_dummy_scale = np.linspace(min(dummy_scale_200), max(dummy_scale_200), 500)
smooth_compute_200M = spline(smooth_dummy_scale)
plt.plot(smooth_dummy_scale, smooth_compute_200M, color='black', linewidth=1, linestyle='dashed')
for i, (x, y, m) in enumerate(zip(dummy_scale_200, compute_200M, markers)):
    plt.plot(x, y, marker=m, color=colors[i], markersize=markersize[i], markeredgecolor=softened_colors[i], zorder=10)

#300
dummy_scale_300 = [300-widths[2], 300, 300+widths[2]]
# derivatives = np.array([[0, 0],  # Derivatives before the first point
#                         [0, 0],  # Derivatives at the first point
#                         [0, 0]])
# bpoly = BPoly.from_derivatives(dummy_scale_300, derivatives)
spline = make_interp_spline(dummy_scale_300, compute_300M, k=2)  # k is the degree of the spline
smooth_dummy_scale = np.linspace(min(dummy_scale_300), max(dummy_scale_300), 10)
smooth_compute_300M = spline(smooth_dummy_scale)
#clip the curve to actual max value of best_2
# smooth_compute_300M = [min(x, best_2) for x in smooth_compute_300M]
plt.plot(smooth_dummy_scale, smooth_compute_300M, color='black', linewidth=1, linestyle='dashed')
for i, (x, y, m) in enumerate(zip(dummy_scale_300, compute_300M, markers)):
    plt.plot(x, y, marker=m, color=colors[i], markersize=markersize[i], markeredgecolor=softened_colors[i], zorder=10)

# plt.xscale('log', base=2)  # Set x-axis to log scale with base 2
# Scatter plot of the best points
plt.scatter(scales[0]-widths[0], best_0, marker=markers[0], color=colors[0], s=300*2, edgecolors='black', linewidths=2, zorder=20)
plt.scatter(scales[1], best_1, marker=markers[1], color=colors[1], s=200*2, edgecolors='black', linewidths=2, zorder=20)
plt.scatter(scales[2]+widths[2], best_2, marker=markers[2], color=colors[2], s=200*2, edgecolors='black', linewidths=2, zorder=20)



#add barplot corresponding to each scatter dot in the same color, have same width for each bar
midpoints = [100, 200, 300]

plt.bar(midpoints[0]-widths[0], compute_100M[0]-2.0, widths[0]*0.9, color=softened_colors[0], edgecolor='black', linewidth=0)
plt.bar(midpoints[0], compute_100M[1]-2.0, widths[0]*0.9, color=softened_colors[1], edgecolor='black', linewidth=0)
plt.bar(midpoints[0]+widths[0], compute_100M[2]-2.0, widths[0]*0.9, color=softened_colors[2], edgecolor='black', linewidth=0)

plt.bar(midpoints[1]-widths[1], compute_200M[0]-2.0, widths[1]*0.9, color=softened_colors[0], edgecolor='black', linewidth=0)
plt.bar(midpoints[1], compute_200M[1]-2.0, widths[1]*0.9, color=softened_colors[1], edgecolor='black', linewidth=0)
plt.bar(midpoints[1]+widths[1], compute_200M[2]-2.0, widths[1]*0.9, color=softened_colors[2], edgecolor='black', linewidth=0)

plt.bar(midpoints[2]-widths[2], compute_300M[0]-2.0, widths[2]*0.9, color=softened_colors[0], edgecolor='black', linewidth=0)
plt.bar(midpoints[2], compute_300M[1]-2.0, widths[2]*0.9, color=softened_colors[1], edgecolor='black', linewidth=0)
plt.bar(midpoints[2]+widths[2], compute_300M[2]-2.0, widths[2]*0.9, color=softened_colors[2], edgecolor='black', linewidth=0)


#over ride x ticks with only 3 ticks, which will be A B and C
plt.xticks(scales, methods)
#xticks font size
plt.xticks(fontsize=14)
#set y axis minimum tick to be 5
plt.ylim(bottom=5)


#plt grid
plt.grid(axis='y', linestyle='dashed', alpha=0.5)
# Labels and Title
plt.xlabel('Total Training Samples', fontsize=16)
plt.ylabel('ImageNet ZeroShot Accuracy', fontsize=16)
#add a legend info
# Custom legend entries
legend_elements = [
    Line2D([0], [0], marker='*', color=softened_colors[0], label='10%', markerfacecolor='red', markersize=15, markeredgecolor='black', linewidth=3, markeredgewidth=0),
    Line2D([0], [0], marker='s', color=softened_colors[1], label='30%', markerfacecolor='green', markersize=10, markeredgecolor='black', linewidth=3, markeredgewidth=0),
    Line2D([0], [0], marker='^', color=softened_colors[2], label='40%', markerfacecolor='blue', markersize=10, markeredgecolor='black', linewidth=3, markeredgewidth=0),
]
plt.legend(handles=legend_elements, fontsize=14, title='Data Retained', title_fontsize=14)

#save the plot
plt.savefig('pareto_clip.png', bbox_inches='tight', dpi=300)
plt.savefig('pareto_clip.pdf', bbox_inches='tight')