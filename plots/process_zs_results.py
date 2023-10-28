import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

def func_d(x, a, b):
    d = 0.67144996
    return a*((x**b) - (x-1)**b)/((x)**d)

def func_normal(x, a, b):
    x_1 = x-1
    return a*((x**b) - (x_1)**b)

# def func(x, a, b):

#     c=0.00
#     return a*(x**b) + c
    # return a*np.log(x) + b
    # return a/(x ** c) + b

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
                    # Path("/project_data2/projects/sachingo/utility_project/mediumscale_nofilter"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_top10p"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_10p_to_20p"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_top30p_10prandom"),
                    Path("/home/sachingo/datacomp/logs/logs/mediumscale_nofilter_5x_sac/"),
                    Path("/project_data2/projects/sachingo/utility_project/mediumscale_laion_5x/"),
                    Path("/project_data2/projects/sachingo/utility_project/mediumscale_clipL14_top30_5x/"),
                    Path("/project_data2/projects/sachingo/utility_project/mediumscale_clipL14_top40p_5x/"),
                    Path("/home/sachingo/datacomp/logs/logs/mediumscale_tmars_5x/"),
                    Path("/project_data2/projects/sachingo/utility_project/mediumscale_dfn_5x"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_30p_to_40p/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_40p_to_50p/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_50p_to_60p/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom_rand10p/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom/"),
                    # Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/wrong_runs/clipbucket_30p_to_40p/"),
                    ]

#make a label_list which is basically last part of the name of folders, extract the last part of the name
label_list = [str(folder_path).split("/")[-1] for folder_path in folder_path_list]
# label_list[-1]   = "whole data random sampling"
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
y_values_original_list = []



for k in range(len(folder_path_list)):
    #run the below code for each k
    folder_path = folder_path_list[k]
    result_dict = {}
    for jsonl_file in folder_path.glob("*.jsonl"):
        # Extract step number from file name
        if k<1:
            match = re.search(r'step_(\d+)\.jsonl', str(jsonl_file))
            if match:
                step_number = int(match.group(1))            
                # if step_number == -1:
                #     continue
                # Open and read jsonl file
                with open(jsonl_file, 'r') as f:
                    avg_score = 0
                    total_entries = 0
                    for line in f:
                        data = json.loads(line)
                        #take average of main metric for each key
                        # if data.get("key") == "imagenet1k":
                        main_metric = data["metrics"]["main_metric"]
                        avg_score += main_metric
                        total_entries += 1
                    assert total_entries == 18, f"total entries is not 18, {jsonl_file}"
                    avg_score = avg_score/total_entries
                    result_dict[step_number] = avg_score
                        
                        # if data.get("key") == "imagenet1k":
                        #     main_metric = data["metrics"]["main_metric"]
                        #     result_dict[step_number] = main_metric
                        #     break
        else:
            #extract epochs numbers from eval_results_epoch_10_step_-1.jsonl
            
            match = re.search(r'epoch_(\d+)_step_-1\.jsonl', str(jsonl_file))
            if match:
                epoch_number = int(match.group(1))            
                # if step_number == -1:
                #     continue
                # Open and read jsonl file
                with open(jsonl_file, 'r') as f:
                    avg_score = 0
                    total_entries = 0
                    for line in f:
                        data = json.loads(line)
                        #take average of main metric for each key
                        # if data.get("key") == "imagenet1k":
                        main_metric = data["metrics"]["main_metric"]
                        if data.get("key") != "misc/winogavil":
                            avg_score += main_metric
                            total_entries += 1
                    assert total_entries == 18, f"total entries is not 18, {jsonl_file}"
                    avg_score = avg_score/total_entries
                    result_dict[epoch_number] = avg_score
                        # if data.get("key") == "imagenet1k":
                        #     main_metric = data["metrics"]["main_metric"]
                        #     result_dict[epoch_number] = main_metric
                        #     break

    # if k==3:
    
    result_dict = {k: v for k, v in sorted(result_dict.items())}
    pruned_result_dict = result_dict
    # pruned_keys = []
    # keys = list(result_dict.keys())
    # for i in range(len(keys)):
    #     if i == 0:
    #         pruned_keys.append(keys[i])
    #     else:
    #         if keys[i] - keys[i-1] > 10:
    #             pruned_keys.append(keys[i])
    # pruned_result_dict = {k: result_dict[k] for k in pruned_keys}
    #divide the keys by the maximum key value
    try:
        max_key = max(pruned_result_dict.keys())
    except:
        import pdb;pdb.set_trace()
    pruned_result_dict = {k/max_key: v for k, v in pruned_result_dict.items()}
    print(pruned_result_dict)
    all_results.append(pruned_result_dict)


for k in range(len(folder_path_list)):
    result_dict = all_results[k]
    x_values = list(result_dict.keys())
    # print(x_values)
    lr_values = [cosine_lr(x_values[i],5e-4,500,128000000/4096) for i in range(len(x_values))]
    x_values = [x_values[i]*4096/1_000_000 for i in range(len(x_values))]
    y_values = list(result_dict.values())
    x_values = x_values[2::2]
    
    y_values = y_values[1:]
    #take mean of every 2 values
    y_values = [(y_values[i]+y_values[i+1])/2 for i in range(0,len(y_values)-1,2)]
    
    x_values = x_values[:-2]
    y_values = y_values[:-2]
    delta_y_values = [y_values[i] - y_values[i - 1] for i in range(1, len(y_values))]
    
    x_values = [i for i in range(1, len(x_values)+1)]

    if k!=0:
        func = func_d
    else:
        func = func_normal

    # params, _ = curve_fit(func, x_values[1:], delta_y_values)
    params = [1,1]
    # print(x_values)
    print("Estimated params after normalizing by repeatings", params)
    delta_y_values_smooth = func(np.array(x_values), *params)
    # import pdb;pdb.set_trace()
    #get values as cumulative sum
    y_values_smooth = np.cumsum(delta_y_values_smooth)
    y_values_original = y_values
    y_values = y_values_smooth
    

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
    y_values_original_list.append(y_values_original)


# Plotting
plt.figure(figsize=(25, 5))
#make a marker list
marker_list = ['o','x','^','s','*','+','D','v','p','h']
#make a color list
color_list = ['b','g','r','c','m','y','k','w']

plt.subplot(1, 5, 1)
for k in range(len(folder_path_list)):
    # plt.plot(x_values_list[k], y_values_list[k], label=label_list[k], color=color_list[k])
    x_vals = list(all_results[k].keys())
    #multiply each elemetn in xvals by by 640
    x_vals = [0] + [x_vals[i]*640 for i in range(len(x_vals))]
    y_vals = [0] + list(all_results[k].values())

    #reduce marker size as well
    marker_width = 2
    plt.plot(x_vals, y_vals, marker=marker_list[k], color=color_list[k], label=label_list[k], linewidth=2, markersize=marker_width)
plt.title('Accuracy vs Number of Samples')
plt.xlabel('Number of Samples (in M)')
plt.ylabel('Average Accuracy on 18 tasks')
plt.legend()
plt.vlines(128, 0, 1, colors='k', linestyles='dashed', label='128M')
#set yaxis limit to 0.4
plt.ylim(0.0, 0.5)
plt.xlim(0, 580)
plt.subplot(1, 5, 2)
for k in range(len(folder_path_list)):
    # plt.plot(x_values_list[k], y_values_list[k], label=label_list[k], color=color_list[k])
    y_vals = all_results[k].values()
    #convert to list
    y_vals = list(y_vals)

    #consecutive diffs 
    x_vals = all_results[k].keys()
    x_vals = list(x_vals)

    y_vals = [(y_vals[i] - y_vals[i-1]) for i in range(1, len(y_vals))]
    plt.scatter(x_vals[2:], y_vals[1:], marker=marker_list[k], color=color_list[k], label=label_list[k])
plt.title('Utility vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.legend()

# plt.subplot(1, 5, 2)
# for k in range(len(folder_path_list)):
#     plt.plot(x_values_list[k][1:], utility_list[k], label=label_list[k], marker=marker_list[k])
# plt.title('Utility vs Number of Samples')
# plt.xlabel('Number of Samples')
# plt.ylabel('Utility')
# plt.legend()
# plt.subplot(1, 5, 3)
# for k in range(len(folder_path_list)):
#     # Smoothing the curve
#     params, _ = curve_fit(func, x_values_list[k][1:], utility_list[k])
    
#     # plt.plot(x_values_list[k][1:], func(np.array(x_values_list[k][1:]), *params), label=label_list[k]+f"a = {params[0]:.2e}, b = {params[1]:.2e}, c = {params[2]:.2e}", color=color_list[k])
#     plt.plot(x_values_list[k][1:], func(np.array(x_values_list[k][1:]), *params), label=label_list[k]+f"a = {params[0]:.2e}, b = {params[1]:.2e}", color=color_list[k])
#     plt.scatter(x_values_list[k][1:], utility_list[k], marker=marker_list[k], color=color_list[k])
#     print(params)
#     # x_new = np.linspace(min(x_values_list[k][1:]), max(x_values_list[k][1:]), 300)
#     # spl = make_interp_spline(x_values_list[k][1:], utility_list[k], k=2)
#     # y_new = spl(x_new)
#     # plt.plot(x_new, y_new, label=label_list[k], marker=marker_list[k])
# plt.title('Smoothened Utility vs Number of Samples')
# plt.xlabel('Number of Samples')
# plt.ylabel('Smoothened Utility')
# plt.legend()

# plt.subplot(1, 5, 4)
# for k in range(len(folder_path_list)):
#     # plt.plot(x_values_list[k][1:-3], utility_by_lr_list[k][:-3], label=label_list[k], marker=marker_list[k])
#     plt.plot(x_values_list[k][1:-1], utility_by_error_list[k][:-1], label=label_list[k], marker=marker_list[k])
# # plt.plot(x_values[1:-3], utility_by_lr[:-3], marker='x')
# # plt.plot(x_values2[1:-3], utility_by_lr2[:-3], marker='x')
# plt.title('Utility by error vs Number of Samples')
# plt.xlabel('Number of Samples')
# plt.ylabel('Utility by lr')
# plt.legend()

# plt.subplot(1, 5, 5)
# for k in range(len(folder_path_list)):
#     plt.plot(x_values_list[k], lr_values_list[k], label=label_list[k], marker=marker_list[k])
# # plt.plot(x_values, lr_values, marker='x')
# # plt.plot(x_values2, lr_values2, marker='x')
# plt.title('LR vs Number of Samples')
# plt.xlabel('Number of Samples')
# plt.ylabel('lr')
# plt.legend()


# plt.savefig('all_utilities_log.png')
plt.savefig('all_5x_results_avg.png')