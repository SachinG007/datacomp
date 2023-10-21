import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
def func(x, a, b):
    c=0.5
    return a/(x ** c) + b

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(step, base_lr, warmup_length, steps):
    if step < warmup_length:
        lr = _warmup_lr(base_lr, warmup_length, step)
    else:
        e = step - warmup_length
        es = steps - warmup_length
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
    return lr

#take a list of folder paths like Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom/") 
folder_path_list = [
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_30p_to_40p/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_40p_to_50p/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_50p_to_60p/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_30p_rand10p/"),
                    ]

#make a label_list which is basically last part of the name of folders, extract the last part of the name
label_list = [str(folder_path).split("/")[-1] for folder_path in folder_path_list]
print(label_list)
all_results = []
x_values_list = []
y_values_list = []
delta_y_values_list = []
delta_x_values_list = []
utility_list = []
utility_by_lr_list = []
lr_values_list = []
utility_by_error_list = []



for k in range(len(folder_path_list)):
    #run the below code for each k
    folder_path = folder_path_list[k]
    result_dict = {}
    for jsonl_file in folder_path.glob("*.jsonl"):
        # Extract step number from file name
        match = re.search(r'step_(\d+)\.jsonl', str(jsonl_file))
        if match:
            step_number = int(match.group(1))            
            if step_number == -1:
                continue
            # Open and read jsonl file
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("key") == "imagenet1k":
                        main_metric = data["metrics"]["main_metric"]
                        result_dict[step_number] = main_metric
                        break
    result_dict = {k: v for k, v in sorted(result_dict.items())}
    pruned_keys = []
    keys = list(result_dict.keys())
    for i in range(len(keys)):
        if i == 0:
            pruned_keys.append(keys[i])
        else:
            if keys[i] - keys[i-1] > 10:
                pruned_keys.append(keys[i])
    pruned_result_dict = {k: result_dict[k] for k in pruned_keys}
    all_results.append(pruned_result_dict)


for k in range(len(folder_path_list)):
    result_dict = all_results[k]
    x_values = list(result_dict.keys())
    print(x_values)
    lr_values = [cosine_lr(x_values[i],5e-4,500,128000000/4096) for i in range(len(x_values))]
    x_values = [x_values[i]*4096/1_000_000 for i in range(len(x_values))]
    y_values = list(result_dict.values())
    # Compute change in accuracy (Delta accuracy)
    delta_y_values = [y_values[i] - y_values[i - 1] for i in range(1, len(y_values))]
    delta_x_values = [x_values[i] - x_values[i - 1] for i in range(1, len(x_values))]
    utility_by_error = [delta_y_values[i]/(0.1-y_values[i]) for i in range(len(delta_y_values))]
    utility = [delta_y_values[i]/delta_x_values[i] for i in range(len(delta_y_values))]
    utility = [max(0, x) for x in utility]
    utility_by_lr = [2*5e-4*utility[i]/(lr_values[i]+lr_values[i+1]) for i in range(len(delta_y_values))]
    #append all the lists to the global list
    x_values_list.append(x_values)
    y_values_list.append(y_values)
    delta_y_values_list.append(delta_y_values)
    delta_x_values_list.append(delta_x_values)
    utility_list.append(utility)
    utility_by_lr_list.append(utility_by_lr)
    lr_values_list.append(lr_values)
    utility_by_error_list.append(utility_by_error)

# Plotting
plt.figure(figsize=(25, 5))
#make a marker list
marker_list = ['o','x','^','s','*','+','D','v','p','h']
#make a color list
color_list = ['b','g','r','c','m','y','k','w']

plt.subplot(1, 5, 1)
for k in range(len(folder_path_list)):
    plt.plot(x_values_list[k], y_values_list[k], label=label_list[k], marker=marker_list[k])
# plt.plot(x_values, y_values, marker='o')
# plt.plot(x_values2, y_values2, marker='x')
plt.title('Accuracy vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 5, 2)
for k in range(len(folder_path_list)):
    plt.plot(x_values_list[k][1:], utility_list[k], label=label_list[k], marker=marker_list[k])
plt.title('Utility vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Utility')
plt.legend()

plt.subplot(1, 5, 3)
for k in range(len(folder_path_list)):
    # Smoothing the curve
    params, _ = curve_fit(func, x_values_list[k][1:], utility_list[k])
    
    # plt.plot(x_values_list[k][1:], func(np.array(x_values_list[k][1:]), *params), label=label_list[k]+f"a = {params[0]:.2e}, b = {params[1]:.2e}, c = {params[2]:.2e}", color=color_list[k])
    plt.plot(x_values_list[k][1:], func(np.array(x_values_list[k][1:]), *params), label=label_list[k]+f"a = {params[0]:.2e}, b = {params[1]:.2e}", color=color_list[k])
    plt.scatter(x_values_list[k][1:], utility_list[k], marker=marker_list[k], color=color_list[k])
    print(params)
    # x_new = np.linspace(min(x_values_list[k][1:]), max(x_values_list[k][1:]), 300)
    # spl = make_interp_spline(x_values_list[k][1:], utility_list[k], k=2)
    # y_new = spl(x_new)
    # plt.plot(x_new, y_new, label=label_list[k], marker=marker_list[k])
plt.title('Smoothened Utility vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Smoothened Utility')
plt.legend()

plt.subplot(1, 5, 4)
for k in range(len(folder_path_list)):
    # plt.plot(x_values_list[k][1:-3], utility_by_lr_list[k][:-3], label=label_list[k], marker=marker_list[k])
    plt.plot(x_values_list[k][1:-1], utility_by_error_list[k][:-1], label=label_list[k], marker=marker_list[k])
# plt.plot(x_values[1:-3], utility_by_lr[:-3], marker='x')
# plt.plot(x_values2[1:-3], utility_by_lr2[:-3], marker='x')
plt.title('Utility by error vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Utility by lr')
plt.legend()

plt.subplot(1, 5, 5)
for k in range(len(folder_path_list)):
    plt.plot(x_values_list[k], lr_values_list[k], label=label_list[k], marker=marker_list[k])
# plt.plot(x_values, lr_values, marker='x')
# plt.plot(x_values2, lr_values2, marker='x')
plt.title('LR vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('lr')
plt.legend()


plt.savefig('all_utilities.png')