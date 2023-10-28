import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

def a_at_every_epoch(a, half_life, max_epochs=30):
    base = 0.5
    a_list = []
    for i in range(max_epochs):
        a_ = a * base**(i/half_life)
        a_list.append(a_)
    return a_list

def data_util_decrease_factor_2(max_epochs=30):
    data_util_list = []
    b = 0.9
    for i in range(max_epochs):
        data_util = (i+1)**b - i**b
        data_util_list.append(data_util)
    return data_util_list

def func_d(x, a, half_life):#, b):
    # d = 0.67144996
    # d = 0.47144996
    # half_life = 7  
    b = 0.9
    # base = (1/np.exp(1))
    base = 0.5
    # x_ = x * base**(x/half_life)
    # x_1 = (x-1) * base**((x-1)/half_life)
    x_ = x
    x_1 = x-1
    # return a*((x_**b) - (x_1)**b)/((x)**d)
    a_ = a * base**(x_1/half_life)
    return a_*((x_**b) - (x_1)**b)

def func_normal(x, a, b):
    b = 0.9
    # c = 10
    x_1 = (x - 1)
    # x = x*c
    return a*((x**b) - (x_1)**b)

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
                    Path("/project_data2/projects/sachingo/utility_project/mediumscale_nofilter"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_top10p"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_10p_to_20p"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_top30p_10prandom"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_30p_to_40p/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_40p_to_50p/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_50p_to_60p/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom_rand10p/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom/"),
                    Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/wrong_runs/clipbucket_30p_to_40p/"),
                    ]

#make a label_list which is basically last part of the name of folders, extract the last part of the name
label_list = [str(folder_path).split("/")[-1] for folder_path in folder_path_list]
label_list[-1]   = "whole data random sampling"
print(label_list)

all_results_dict = {}
all_results = []
all_results = []
x_values_list = []
y_values_list = []
delta_y_values_list = []
delta_x_values_list = []
utility_list = []
utility_by_lr_list = []
lr_values_list = []
utility_by_error_list = []
f_k_list = []
estimated_k_list = []
utility_by_error_list = []
y_values_original_list = []

for k in range(len(folder_path_list)):
    #run the code for each k
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
    lr_values = [cosine_lr(x_values[i],5e-4,500,128000000/4096) for i in range(len(x_values))]
    x_values = [x_values[i]*4096/1_000_000 for i in range(len(x_values))]
    y_values = list(result_dict.values())
    y_values = [1 - y_values[i] for i in range(len(y_values))]
    x_values = x_values[2::2]
    
    y_values = y_values[1:]
    #take mean of every 2 values
    y_values = [(y_values[i]+y_values[i+1])/2 for i in range(0,len(y_values),2)]
    
    x_values = x_values[:-2]
    y_values = y_values[:-2]

    
    #append 0 to the beginning of y_values
    appended_y_values = [1] + y_values
    delta_y_values = [appended_y_values[i-1] - appended_y_values[i] for i in range(1, len(appended_y_values))]
    
    x_values = [i for i in range(1, len(x_values)+1)]

    if k!= 0:
        func = func_d
    else:
        func = func_normal

    params, _ = curve_fit(func, x_values, delta_y_values)
    # params = [1, 0.1]
    # print(x_values)
    print(label_list[k])
    print("Estimated params after normalizing by repeatings", params)
    #half life at each epoch
    
    a_at_every_epoch_list = a_at_every_epoch(params[0], params[1])
    #print the list values separated by comma
    
    # for j in range(len(a_at_every_epoch_list)):
    #     print(a_at_every_epoch_list[j], end=", ")
    
    # print("\n")
    # import pdb; pdb.set_trace()
    delta_y_values_smooth = func(np.array(x_values), *params)
    #get values as cumulative sum
    y_values_smooth = 1 - np.cumsum(delta_y_values_smooth)
    y_values_original = y_values
    y_values = y_values_smooth
    # import ipdb; ipdb.set_ trace()

    y_values_list.append(y_values)
    y_values_original_list.append(y_values_original)
    x_values_list.append(x_values)
    

# Plotting
plt.figure(figsize=(5, 5))
#make a marker list
marker_list = ['o','x','^','s','*','+','D','v','p','h']
#make a color list
color_list = ['b','g','r','c','m','y','k','purple','orange','pink']

for k in range(len(folder_path_list)):
    plt.plot(x_values_list[k], y_values_list[k], label=label_list[k], color=color_list[k])
    plt.scatter(x_values_list[k], y_values_original_list[k], marker=marker_list[k], color=color_list[k])
plt.title('Accuracy vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('all_utilities_pm.png')

data_util_decrease_factor_2_list = data_util_decrease_factor_2()
#print the list elements separated by comma
# for j in range(len(data_util_decrease_factor_2_list)):
#     print(data_util_decrease_factor_2_list[j], end=", ")