import matplotlib.pyplot as plt
import json
import re
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import differential_evolution, curve_fit

from optim_bkp import CustomOptimizer

def power(x, power):
    if x==0:
        return 0
    return x**power

def func_old(samples_seen, params, samples_per_epoch, normalizer = 1_000_000):
    base = 0.5
    num_epochs_full = int(samples_seen//samples_per_epoch)
    a,b,c,d = params

    effective_samples = 0
    for i in range(num_epochs_full):
        effective_samples += samples_per_epoch * base**(i*c)
    remaining_samples = samples_seen - effective_samples
    effective_samples += remaining_samples * base**(num_epochs_full*c)

    acc = a*power(effective_samples,b)# + d
    return acc

def func(samples_seen, params, samples_per_epoch, normalizer = 1_000_000):
    base = 0.5
    num_epochs_full = int(samples_seen//samples_per_epoch)
    a,b,c,d = params

    #sum up the utilities per epoch
    acc = 0+d
    for i in range(num_epochs_full + 1):
        if i != num_epochs_full:
            samples = samples_per_epoch * (i+1)
            samples_1 = samples_per_epoch * i
        else:
            samples = samples_seen
            samples_1 = samples_per_epoch * i
                
        samples, samples_1 = samples/normalizer, samples_1/normalizer
        acc += a * (power(samples,b) - power(samples_1,b))*base**(i*c*(-b))
    
    return acc



from all_laion import samples_per_epoch_dict, paths_b32, paths_l14, match_with_dict, subsample_every_dict, paths_b16
# from all_paths_nofilter import samples_per_epoch_dict, paths, match_with_dict, subsample_every_dict
paths = paths_b16

def get_accuracy_from_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("key") == "imagenet1k":
                main_metric = data["metrics"]["main_metric"]
    return main_metric


def get_all_results_from_folder(data_name, subsample_every=None):
    folder_path = paths[data_name]
    result_dict = {}
    for key in folder_path.keys():
        jsonl_file = folder_path[key]
        #key = 3B. convert this into 3_000_000_000 and so on
        step = int(key[:-1])*1_000_000_000
        result_dict[step] = get_accuracy_from_jsonl(jsonl_file)

    result_dict = {k: v for k, v in sorted(result_dict.items())}

    return result_dict
    
all_results = {}
# remove 80M key from paths
# paths.pop("2B")
# paths.pop("400M")

for key in paths.keys():
    all_results[key] = (get_all_results_from_folder(key, subsample_every = subsample_every_dict[key]))


x_vals = []
y_vals = []
error_vals = []
samples_per_epoch_vals = []

# create a common input for all the data
for key in paths.keys():
    curent_dict = all_results[key]
    samples_per_epoch  = samples_per_epoch_dict[key] 
    for step in curent_dict.keys():
        x_vals.append(step)
        y_vals.append(curent_dict[step])
        error_vals.append( 1 - curent_dict[step])
        samples_per_epoch_vals.append(samples_per_epoch)



def get_params_from_data(x_vals, samples_per_epoch_vals, error_vals):
    optimizer = CustomOptimizer(x_vals, error_vals, samples_per_epoch_vals, 4, func, 0.001, 500)
    popt = optimizer.optimize()
    return popt


popt = get_params_from_data(x_vals, samples_per_epoch_vals, error_vals)
popt = [x.item() for x in popt]
# a: 0.3425 | b: -0.0680 | c: 0.7617
# b-32 - p: 0.9856536388397217 p: -0.10479629784822464 p: 38.90073776245117 
# b-16 - p: 0.981525719165802 p: -0.11960593611001968 p: 35.33066177368164 
# popt  = [0.3425, -0.0680, 2]

# popt = [0.981525719165802, -0.11960593611001968, 35.33066177368164]
# popt = [0.9856536388397217, -0.10479629784822464, 38.90073776245117]
def get_fit_dict(popt):
    fit_dict = {}
    for key in paths.keys():
        curent_dict = all_results[key]
        samples_per_epoch  = samples_per_epoch_dict[key]
        fit = [] 
        for step in curent_dict.keys():
            fit.append(func(step, popt, samples_per_epoch))
        fit_dict[key] = fit 
    return fit_dict

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple', 'brown', 'gray']
markers = ['o', 'x', '^', 's', 'p', '*', '+', 'D', 'v', 'h', '8', '1']



def plot_fit_dict(fit_dict, all_results):
    for i, key in enumerate(paths.keys()):
        errors = [1 - x for x in all_results[key].values()]
        plt.scatter(list(all_results[key].keys()), errors, label=key, color=colors[i], marker=markers[i])
        plt.plot(list(all_results[key].keys()), fit_dict[key], label=key, color=colors[i])
    # legend outside to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("laion_with_constant_sac.png", bbox_inches='tight')


# popt = [0.9856536388397217, -0.10479629784822464, 38.90073776245117]
def do_all(popt):
    plt.clf()
    fit_dict = get_fit_dict(popt=popt)
    plot_fit_dict(fit_dict, all_results)
    return fit_dict

# import pdb; pdb.set_trace()
#  0.979049026966095 p: -0.12033262103796005 p: 77.01390838623047
# popt = [0.979049026966095, -0.12033262103796005, 77.01390838623047]
do_all(popt)
#line plot fitted curve


