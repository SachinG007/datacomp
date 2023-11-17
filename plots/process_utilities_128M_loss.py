import matplotlib.pyplot as plt
import json
import re
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import differential_evolution, curve_fit

from optim import CustomOptimizer
import torch

def power(x, power):
    if x==0:
        return 0
    return x**power

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
        acc += a * (power(samples,b) - power(samples_1,b))*base**(i/c)
    
    return acc


from all_paths_128 import samples_per_epoch_dict, paths, match_with_dict, subsample_every_dict
# from all_paths_nofilter import samples_per_epoch_dict, paths, match_with_dict, subsample_every_dict

samples_per_step = 4096

def get_accuracy_from_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("key") == "imagenet1k":
                main_metric = data["metrics"]["main_metric"]
    return main_metric


def get_all_results_from_folder(data_name, subsample_every=None):
    folder_path = paths[data_name]
    match_with = match_with_dict[data_name]
    result_dict = {}
    for jsonl_file in folder_path.glob("*.jsonl"):
        if match_with == "step" :
            match = re.search(r'step_(\d+)\.jsonl', str(jsonl_file))
            if match:
                step_number = int(match.group(1))            
                if step_number == -1 or step_number == 0:
                    continue
                result_dict[step_number*samples_per_step] = get_accuracy_from_jsonl(jsonl_file)

        if match_with == "epoch" :
            # extract epoch from eval_results_epoch_10_step_-1.jsonl 
            match = re.search(r'epoch_(\d+)_', str(jsonl_file))
            if match:
                epoch_number = int(match.group(1)) 
                step_number = epoch_number * samples_per_epoch_dict[data_name] / samples_per_step           
                result_dict[step_number*samples_per_step] = get_accuracy_from_jsonl(jsonl_file)

    result_dict = {k: v for k, v in sorted(result_dict.items())}
    pruned_keys = []
    keys = list(result_dict.keys())
    for i in range(len(keys)):
        if i == 0:
            pruned_keys.append(keys[i])
        else:
            if keys[i] - keys[i-1] > 10*samples_per_step:
                pruned_keys.append(keys[i])
    pruned_result_dict = {k: result_dict[k] for k in pruned_keys}

    if subsample_every is not None:
        #take the average of every subsample_every values
        pruned_result_dict = {k: np.mean(list(pruned_result_dict.values())[i:i+subsample_every]) for i, k in enumerate(pruned_result_dict.keys()) if i % subsample_every == 0}
    return pruned_result_dict
    
all_results = {}
for key in paths.keys():
    all_results[key] = (get_all_results_from_folder(key, subsample_every = subsample_every_dict[key]))


x_vals_dict = {}
error_vals_dict = {}
y_vals_dict = {}

for key in paths.keys():
    x_vals_dict[key] = list(all_results[key].keys())
    y_vals = list(all_results[key].values())
    y_vals_dict[key] = y_vals
    error_vals_dict[key] = [1 - y_vals[i] for i in range(len(y_vals))]

def get_params_from_data(data_name):
    error_vals = error_vals_dict[data_name]
    y_vals = y_vals_dict[data_name]
    x_vals = x_vals_dict[data_name]
    samples = samples_per_epoch_dict[data_name]
    samples = [samples for i in range(len(x_vals))]
    optimizer = CustomOptimizer(x_vals, error_vals, samples,  4, func, 0.001, 1000)
    popt = optimizer.optimize()
    return popt



fitted_vals_dict = {}
for key in paths.keys():
    print("******", key)
    popt = get_params_from_data(key)
    x_vals = x_vals_dict[key]
    samples = samples_per_epoch_dict[key]

    fit = []
    for i in range(len(x_vals)):
        # fit.append(func(x_vals[i], popt, samples).item())
        fit.append(func(x_vals[i], popt, samples).item())
    fitted_vals_dict[key] = fit
# import pdb; pdb.set_trace()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple', 'brown', 'gray']
markers = ['o', 'x', '^', 's', 'p', '*', '+', 'D', 'v', 'h', '8', '1']


# for i, key in enumerate(paths.keys()):
#     data_name = key
#     x_vals = x_vals_dict[data_name]
#     delta_y_vals = error_vals_dict[data_name]
#     #y_vals is the cumulative sum of delta_y_vals
#     y_vals = (delta_y_vals)
#     plt.scatter(x_vals, y_vals, label=data_name, color=colors[i], marker=markers[i])
#     fitted_deltas = fitted_vals_dict[data_name]
#     fitted = fitted_deltas
#     plt.plot(x_vals, fitted, color=colors[i])
#     # legend outside to the right
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.savefig("temp_128_acc.png", bbox_inches='tight')

for i, key in enumerate(paths.keys()):
    data_name = key
    x_vals = x_vals_dict[data_name]
    y_vals = y_vals_dict[data_name]
    error_vals = error_vals_dict[data_name]
    plt.scatter(x_vals, error_vals, label=data_name, color=colors[i], marker=markers[i])
    fitted_deltas = fitted_vals_dict[data_name]
    fitted = fitted_deltas
    plt.plot(x_vals, fitted, color=colors[i])
    # legend outside to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig("temp_128_acc.png", bbox_inches='tight')
