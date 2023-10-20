import json
import re
from pathlib import Path
import numpy as np

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

folder_path = Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_60p_to_bottom/")  # Replace with your folder path
result_dict = {}

for jsonl_file in folder_path.glob("*.jsonl"):
    # Extract step number from file name
    match = re.search(r'step_(\d+)\.jsonl', str(jsonl_file))
    if match:
        step_number = int(match.group(1))
        
        # Open and read jsonl file
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("key") == "imagenet1k":
                    main_metric = data["metrics"]["main_metric"]
                    result_dict[step_number] = main_metric
                    break

print(result_dict)

folder_path = Path("/project_data2/projects/sachingo/utility_project/clip_bucket_training/clipbucket_30p_to_40p/")  # Replace with your folder path
result_dict2 = {}

for jsonl_file in folder_path.glob("*.jsonl"):
    # Extract step number from file name
    match = re.search(r'step_(\d+)\.jsonl', str(jsonl_file))
    if match:
        step_number = int(match.group(1))
        
        # Open and read jsonl file
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("key") == "imagenet1k":
                    main_metric = data["metrics"]["main_metric"]
                    result_dict2[step_number] = main_metric
                    break

print(result_dict2)

import matplotlib.pyplot as plt

# Sort the dictionary by step number (number of samples)
sorted_dict2 = {k: v for k, v in sorted(result_dict2.items())}

# Prepare data for plotting
x_values2 = list(sorted_dict2.keys())
print(x_values2)
lr_values2 = [cosine_lr(x_values2[i],5e-4,500,128000000/4096) for i in range(len(x_values2))]

x_values2 = [x_values2[i]*4096/1_000_000 for i in range(len(x_values2))]
y_values2 = list(sorted_dict2.values())

# Compute change in accuracy (Delta accuracy)
delta_y_values2 = [y_values2[i] - y_values2[i - 1] for i in range(1, len(y_values2))]
delta_x_values2 = [x_values2[i] - x_values2[i - 1] for i in range(1, len(x_values2))]

utility2 = [delta_y_values2[i]/delta_x_values2[i] for i in range(len(delta_y_values2))]
utility2 = [max(0, x) for x in utility2]
utility_by_lr2 = [2*5e-4*utility2[i]/(lr_values2[i]+lr_values2[i+1]) for i in range(len(delta_y_values2))]




# Sort the dictionary by step number (number of samples)
sorted_dict = {k: v for k, v in sorted(result_dict.items())}

# Prepare data for plotting
x_values = list(sorted_dict.keys())
print(x_values)
lr_values = [cosine_lr(x_values[i],5e-4,500,128000000/4096) for i in range(len(x_values))]

x_values = [x_values[i]*4096/1_000_000 for i in range(len(x_values))]
y_values = list(sorted_dict.values())

# Compute change in accuracy (Delta accuracy)
delta_y_values = [y_values[i] - y_values[i - 1] for i in range(1, len(y_values))]
delta_x_values = [x_values[i] - x_values[i - 1] for i in range(1, len(x_values))]

utility = [delta_y_values[i]/delta_x_values[i] for i in range(len(delta_y_values))]

utility_by_lr = [2*5e-4*utility[i]/(lr_values[i]+lr_values[i+1]) for i in range(len(delta_y_values))]



# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
print(y_values)
print(y_values2)
plt.plot(x_values, y_values, marker='o')
plt.plot(x_values2, y_values2, marker='x')
plt.title('Accuracy vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')

plt.subplot(1, 4, 2)
plt.plot(x_values[1:], utility, marker='x')
plt.plot(x_values2[1:], utility2, marker='x')
plt.title('Utility vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Utility')

plt.subplot(1, 4, 3)
plt.plot(x_values[1:-3], utility_by_lr[:-3], marker='x')
plt.plot(x_values2[1:-3], utility_by_lr2[:-3], marker='x')
plt.title('Utility div LR vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Utility by lr')

plt.subplot(1, 4, 4)
plt.plot(x_values, lr_values, marker='x')
plt.plot(x_values2, lr_values2, marker='x')
plt.title('LR vs Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('lr')


plt.savefig('utility_by_lr_3op_to_40p.png')
# plt.show()