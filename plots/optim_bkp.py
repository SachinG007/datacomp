import numpy as np
import torch
from tqdm import tqdm


def power(x, power):
    if x==0:
        return 0
    return x**power

def func(index, params, x_orig, x_1):
    acc = 0
    base = 0.5
    # a = 1
    # b = -0.0729
    a,b = params
    for i in range(index):
        acc += a * (power(x_orig[i],b) - power(x_1[i],b))
    return acc

class CustomOptimizer:
    def __init__(self, x, y, samples_per_epoch, num_params, optim_func, learning_rate = 100, num_iterations = 10000, verbose = True):
        self.x = x
        self.y = y
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.num_params = num_params
        self.params = []
        self.func = optim_func
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.init_params()

    def init_params(self):
        initialization = [1, -0.01, 0.01, 0.01]
        # p_1 = torch.randn((), requires_grad=True)
        for i in range(self.num_params):
            p = torch.randn((), requires_grad=True)
            p.data = p.data*0 + initialization[i]
            p.grad = p.data*0
            self.params.append(p)
        print("optimizer print: initial params", self.params)
        print("optimizer print: self x", self.x)
        print("optimizer print: self y", self.y)

    def optimize(self):
        pbar = tqdm(total=self.num_iterations)
        for i in range(self.num_iterations):
            # Compute the function value
            loss = 0
            for idx in range(len(self.y)):
                samples_per_epoch = self.samples_per_epoch[idx]
                samples_seen = self.x[idx]
                func_value = self.func(samples_seen, self.params, samples_per_epoch)
                true_value = self.y[idx]
                curr_loss = (func_value - true_value)**2
                loss += curr_loss

            # Compute gradients
            loss.backward()

            # Update variables with gradient descent
            with torch.no_grad():
                for j,p in enumerate(self.params):
                    if j == 2:
                        lr_factor = 10
                    else:
                        lr_factor = 1
                    p -= self.learning_rate * p.grad * lr_factor
                    # Manually zero the gradients after updating weights
                    p.grad.zero_()
                    self.params[j] = p
            
            pbar.update(1)
            pbar.set_description(f'Iteration {i}: loss: {loss.item():.4f} | a: {self.params[0].item():.4f} | b: {self.params[1].item():.4f} | c: {1/self.params[2].item():.4f}  | d: {self.params[3].item():.4f}')
        
        if self.verbose:
            print(f'Final loss: {loss.item()}')
            for p in self.params:
                print(f'p: {p.item()}', end=' ')
            print()

        return [x.item() for x in self.params]
    
    def predict(self, index):
        return self.func(index, self.params, self.x, self.x_1)


        