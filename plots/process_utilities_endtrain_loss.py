import matplotlib.pyplot as plt
import json
import re
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import differential_evolution, curve_fit

def func(inputs, a, b, half_life):
    ep = inputs[0]
    x = inputs[1] / 12_800_000
    x_1 = inputs[2]  / 12_800_000 + 0.001

    base = 0.5
    b = -0.0729
    # half_life = 2.5
    # half_life = 1
    ep = 0
    a_ = a * base**(ep/half_life)
    # import pdb; pdb.set_trace()
    # return a_ * (x**b - x_1**b)
    return a_ * (x**b )

# from all_paths_640 import samples_per_epoch_dict, paths, match_with_dict, subsample_every_dict
from all_paths_nofilter import samples_per_epoch_dict, match_with_dict, subsample_every_dict, paths_16, paths_32, paths_64, paths_128

samples_per_step = 4096
paths = paths_16
def get_accuracy_from_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("key") == "imagenet1k":
                main_metric = data["metrics"]["main_metric"]
    return main_metric


def get_all_results_from_folder(data_name, paths, subsample_every=None):
    folder_path = paths
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
        if match_with == "last_step" :
            match = re.search(r'step_(\d+)\.jsonl', str(jsonl_file))
            if match:
                step_number = int(match.group(1))
                if step_number == -1 or step_number == 0:
                    continue
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
    
all_results_nofilter = {}
all_results = {}
paths_list = [paths_16, paths_32, paths_64, paths_128]
for key in paths_16.keys():
    current_dict = {}
    for path in paths_list:
        results = (get_all_results_from_folder(key, path[key],subsample_every = subsample_every_dict[key]))
        #extract the max key and its value
        max_key = max(results.keys())
        max_val = results[max_key]
        current_dict[max_key] = 1 - max_val
    #add this to the all_results_nofilter dict
    
    all_results[key] = current_dict
print(all_results)
x_vals_dict = {}
x_1_vals_dict = {}
ep_dict = {}
delta_y_vals_dict = {}
inputs_dict = {}

for key in paths_16.keys():
    x_1_vals_dict[key] = [0] + list(all_results[key].keys())[:-1]
    x_vals_dict[key] = list(all_results[key].keys())
    ep_dict[key] = [0] + [x//samples_per_epoch_dict[key] for x in x_vals_dict[key]][:-1]
    y_1_vals = [1] + list(all_results[key].values())[:-1]
    y_vals = list(all_results[key].values())
    # delta_y_vals_dict[key] = [y_1_vals[i] - y_vals[i] for i in range(len(y_vals))]
    delta_y_vals_dict[key] = [y_vals[i] for i in range(len(y_vals))]
    inputs_dict[key] = [ep_dict[key], x_vals_dict[key], x_1_vals_dict[key]]
    print(inputs_dict)
    print(delta_y_vals_dict)
    


def get_params_from_data(data_name):
    y_vals = delta_y_vals_dict[data_name]
    inputs = inputs_dict[data_name]
    popt, _ = curve_fit(func, inputs, y_vals)
    return popt

for key in paths.keys():
    popt = get_params_from_data(key)
    print(key, popt)

# plot the data with y axis as accuracy, x axis as num samples seen. 
# plot scatter for real points and plot curve for fitted curve.


fitted_vals_dict = {}
for key in paths.keys():
    popt = get_params_from_data(key)
    ep_vals = ep_dict[key]
    x_vals = x_vals_dict[key]
    x_1_vals = x_1_vals_dict[key]

    fit = []
    for i in range(len(x_vals)):
        fit.append(func([ep_vals[i], x_vals[i], x_1_vals[i]], *popt))
    fitted_vals_dict[key] = fit

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple', 'brown', 'gray']
markers = ['o', 'x', '^', 's', 'p', '*', '+', 'D', 'v', 'h', '8', '1']


for i, key in enumerate(paths.keys()):
    data_name = key
    x_vals = x_vals_dict[data_name]
    delta_y_vals = delta_y_vals_dict[data_name]
    #y_vals is the cumulative sum of delta_y_vals
    y_vals = np.cumsum(delta_y_vals)
    # y_vals = delta_y_vals

    inputs = inputs_dict[data_name]
    popt = get_params_from_data(data_name)

    plt.scatter(x_vals, y_vals, label=data_name, color=colors[i], marker=markers[i])
    fitted_deltas = fitted_vals_dict[data_name]
    fitted = np.cumsum(fitted_deltas) 
    # fitted = fitted_deltas
    plt.plot(x_vals, fitted, color=colors[i])
    # legend outside to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig("temp.png", bbox_inches='tight')
