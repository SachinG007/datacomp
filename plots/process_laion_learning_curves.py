import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import numpy as np
from optim_bkp import CustomOptimizer
import copy
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

def nonvec_func(samples_seen, params, samples_per_epoch, normalizer = 1_000_000):
    base = 0.5
    num_epochs_full = int(samples_seen//samples_per_epoch)
    a,b,c,d = params

    #sum up the utilities per epoch
    loss = 0+d
    for i in range(num_epochs_full + 1):
        if i != num_epochs_full:
            samples = samples_per_epoch * (i+1)
            samples_1 = samples_per_epoch * i
        else:
            #in case of partial epoch
            samples = samples_seen
            samples_1 = samples_per_epoch * i

        #normalizer to stabilize optimization        
        samples, samples_1 = samples/normalizer, samples_1/normalizer

        #add the marginal change in loss times the diminishing factor (base**(i*c*(-b)))
        loss += a * (power(samples,b) - power(samples_1,b))*base**(i*c*(-b))
    
    return loss

def func(samples_seen, params, samples_per_epoch, normalizer=1_000_000):
    base = 0.5
    num_epochs_full = samples_seen // samples_per_epoch
    a, b, c, d = params

    # Creating arrays for each epoch and an additional one for partial epoch if exists
    epochs = np.arange(num_epochs_full + 1)
    samples = np.minimum(samples_per_epoch * (epochs + 1), samples_seen)
    samples_1 = samples_per_epoch * epochs

    # Normalizing the samples
    samples, samples_1 = samples / normalizer, samples_1 / normalizer
    samples_all = copy.deepcopy(samples)
    
    # Calculating the loss
    loss = d
    if len(samples_all) > 1:
        samples, samples_1, epochs = samples[1:], samples_1[1:], epochs[1:]
        vec_loss = a * (np.power(samples, b) - np.power(samples_1, b)) * base**(epochs/c)
        loss += vec_loss.sum()
    loss += a * (power(samples_all[0], b))

    return loss

def get_params_from_data(x_vals, samples_per_epoch_vals, error_vals):
    optimizer = CustomOptimizer(x_vals, error_vals, samples_per_epoch_vals, 4, func, 0.001, 500)
    popt = optimizer.optimize()
    return popt


def get_params_from_data_grid(x_vals, samples_per_epoch_vals, error_vals):
    a = 1
    b_lim = np.linspace(-0.01, -0.2, 100)
    c_lim = np.linspace(1, 100, 100)
    d_lim = np.linspace(0.01, 0.4, 100)
    import pdb; pdb.set_trace()
    #create a grid of all possible combinations
    grid = np.array(np.meshgrid(a, b_lim, c_lim, d_lim)).T.reshape(-1,4)
    #randomize the grid
    np.random.shuffle(grid)
    #get the best params by running func on all combinations
    #also store the loss grid to plot later
    best_params = None
    best_loss = 10000
    pbar = tqdm(total=len(grid))
    for params in grid:
        loss = 0
        for i in range(len(x_vals)):
            samples_per_epoch = samples_per_epoch_vals[i]
            samples_seen = x_vals[i]
            func_value = func(samples_seen, params, samples_per_epoch)
            true_value = error_vals[i]
            curr_loss = (func_value - true_value)**2
            loss += curr_loss
        # loss_grid.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params

        pbar.update(1)
        pbar.set_description("best loss: {} : best params : {}".format(best_loss, best_params))
    
    #plot the loss grid using only 2 axis b and c, with best a and d values
    print("best params", best_params)
    return best_params

    # optimizer = CustomOptimizer(x_vals, error_vals, samples_per_epoch_vals, 4, func, 0.001, 500)
    # popt = optimizer.optimize()
    # return popt


from all_laion import samples_per_epoch_dict, paths_b32, paths_l14, match_with_dict, subsample_every_dict, paths_b16
# from all_paths_nofilter import samples_per_epoch_dict, paths, match_with_dict, subsample_every_dict
# load the csv
csv_path = "imagenet_zeroshot_learning_curves.csv"
import pandas as pd
df = pd.read_csv(csv_path)



paths = paths_b16



def get_accuracy_from_csv(jsonl_file):
    #get all values corresponding to name jsonl_file in df. change name by replacing jsonl with pt
    name = jsonl_file.replace("jsonl", "pt").replace("/home/sachingo/datacomp/logs/logs/laion-scaling/eval_results_","")
    df_name = df[df["name"] == name]
    #sort by epoch as int (read as int)
    df_name = df_name.sort_values(by=["epoch"], key=lambda x: x.astype(int))
    #get a list of "epochs" and "accuracies"
    epochs = df_name["epoch"].tolist()
    accuracies = df_name["imagenet-zeroshot-val-top1"].tolist()
    return epochs, accuracies


def get_all_results_from_csv(data_name):
    folder_path = paths[data_name]
    result_dict = {}
    for key in folder_path.keys():
        total_samples_seen = int(key[:-1])*1_000_000_000
        jsonl_file = folder_path[key]
        epoch, accuracy = get_accuracy_from_csv(jsonl_file)
        num_ckpt = len(epoch)
        samples_per_ckpt = total_samples_seen/num_ckpt
        num_samples = [x*samples_per_ckpt for x in epoch]
        result_dict[key] = dict(zip(num_samples, accuracy))
    return result_dict
    
all_results = {}
# remove 80M key from paths
# paths.pop("2B")
# paths.pop("400M")

for key in paths.keys():
    all_results[key] = (get_all_results_from_csv(key))

x_vals = []
y_vals = []
error_vals = []
samples_per_epoch_vals = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple', 'brown', 'gray']
markers = ['o', 'x', '^', 's', 'p', '*', '+', 'D', 'v', 'h', '8', '1']

fig = plt.figure()
# create a common input for all the data
for key in paths.keys():
    curent_dict = all_results[key]
    samples_per_epoch  = samples_per_epoch_dict[key] 
    for step in curent_dict.keys():
        res = curent_dict[step]
        x_vals_current = res.keys()
        #convert to list
        x_vals_current = list(x_vals_current)
        y_vals_current = list(res.values())
        #change to error
        y_vals_current = [1 - x for x in y_vals_current]
        samples_per_epoch_vals_current = [samples_per_epoch]*len(x_vals_current)
        params_current = get_params_from_data_grid(x_vals_current, samples_per_epoch_vals_current, y_vals_current)
        #plot the fitted curve and the true scatter
        legend = key + " " + step
        plt.scatter(x_vals_current, y_vals_current, label=legend, color=colors[len(x_vals)], marker=markers[len(x_vals)])
        plt.plot(x_vals_current, [func(x, params_current, samples_per_epoch) for x in x_vals_current], color=colors[len(x_vals)])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig("laion_training.png", bbox_inches='tight')



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


