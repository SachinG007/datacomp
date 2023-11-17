import numpy as np
import torch
import copy
def power(x, power):
    if x==0:
        return 0
    return x**power

def vectorized_func(samples_seen, params, samples_per_epoch, normalizer=1_000_000):
    base = 0.5
    num_epochs_full = samples_seen // samples_per_epoch
    print("num_epochs_full", num_epochs_full)
    import pdb; pdb.set_trace()
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
        vec_loss = a * (np.power(samples, b) - np.power(samples_1, b)) * base**(epochs * c * (-b))
        loss += vec_loss.sum()
    loss += a * (power(samples_all[0], b)) * base**(0 * c * (-b))

    return loss

def func(samples_seen, params, samples_per_epoch, normalizer = 1_000_000):
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

#run the 2 functions on some random set of values
samples_seen = 1_000_001
samples_per_epoch = 1_000_000
params = [1, -0.01, 0.01, 0.01]
print(func(samples_seen, params, samples_per_epoch))
print(vectorized_func(samples_seen, params, samples_per_epoch))
