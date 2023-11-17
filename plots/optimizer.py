import numpy as np
import torch

deltas = [0.033479999999999954, 0.02946000000000004, 0.04815999999999998, 0.059880000000000044]
outs = 1 - np.cumsum(deltas)
x_orig = [16011264, 32026624, 64016384, 127959040]
x_1 = [0, 16011264, 32026624, 64016384]
# x_orig = np.array(x_orig)/12_800_000
# x_1 = np.array(x_1)/12_800_000
x_orig = [x/12_800_000 for x in x_orig]
x_1 = [x/12_800_000 for x in x_1]
inps = np.arange(1,5).tolist()

# Define the function func(x, y, z)
# def func(x, y, z):
#     # Example: func(x, y, z) = e^(x + 2y - z) + z*(y^2)
#     return torch.exp(x + 2 * y - z) + z * (y**2)


def power(x, power):
    if x==0:
        return 0
    return torch.pow(x,power)

# def func(index, a, b, c):
#     acc = 0
#     base = 0.5
#     # a = 1
#     # b = -0.0729
#     for i in range(index):
#         acc += a * (power(x_orig[i],b) - power(x_1[i],b))
#     return acc

def func(index, params, x_orig, x_1):
    acc = 0
    base = 0.5
    # a = 1
    # b = -0.0729
    a,b,c = params
    for i in range(index+1):
        acc += a * (power(x_orig[i],b) - power(x_1[i],b))*base**(0/c)
    return acc

# Initialize variables x, y, z
# x = torch.tensor((1.0), requires_grad=True)
# y = torch.tensor((-0.1), requires_grad=True)
# z = torch.tensor((1.0), requires_grad=True)
# Initialize variables x, y, z
x = torch.randn((), requires_grad=True)
y = torch.randn((), requires_grad=True)
z = torch.randn((), requires_grad=True)

# Set the learning rate
learning_rate = 0.0001

# inps = [[0, 0, 0, 0], [16011264, 32026624, 64016384, 127959040], [0, 16011264, 32026624, 64016384]]
# inps_formatted = []
# for i in range(4):
#     curr_inp = []
#     for j in range(len(inps)):
#         curr_inp.append(inps[j][i])
#     inps_formatted.append(curr_inp)
# print("inps formatted", inps_formatted)
# inps = inps_formatted
                                  
# outs = [-0.033479999999999954-10, -0.02946000000000004, -0.04815999999999998, -0.059880000000000044]

# outs = [-0.02946000000000004, -0.04815999999999998, -0.059880000000000044]
# outs = [0.96652, 0.93706, 0.8889, 0.82902]

# Set the number of iterations
num_iterations = 50000*1

from optim import CustomOptimizer
optimizer = CustomOptimizer(x_orig, outs, 3, func, 0.001, 50000)
popt = optimizer.optimize()


# for i in range(num_iterations):
#     # Compute the function value
#     loss = 0
#     for inp,out in zip(inps, outs):
#         function_value = func(inp,x, y, z)
#         true_value = out
#         loss += (function_value - true_value)**2

#     # Compute gradients
#     loss.backward()

#     # Update variables with gradient descent
#     with torch.no_grad():
#         x -= learning_rate * x.grad
#         y -= learning_rate * y.grad
#         # z -= learning_rate * z.grad

#         # Manually zero the gradients after updating weights
#         x.grad.zero_()
#         y.grad.zero_()
#         # z.grad.zero_()

#     # Print the function value at each step or at certain intervals
#     if i % 100 == 0:
#         print(f'Iteration {i}: loss: {loss.item()}')
#         #print values of x, y, z in one line
#         print(f'x: {x.item()}, y: {y.item()}, z: {z.item()}')

# # Print the optimized variables and the minimum function value
# print(f'Optimized x: {x.item()}')
# print(f'Optimized y: {y.item()}')
# print(f'Optimized z: {z.item()}')
# # print(f'Minimum func(x, y, z): {func(x, y, z).item()}')