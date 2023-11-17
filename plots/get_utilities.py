# mediumscale_nofilter
# Estimated params after normalizing by repeatings [0.02400279 1.        ]
# clipbucket_top10p
# Estimated params after normalizing by repeatings [0.04672413 3.74046299]
# clipbucket_10p_to_20p
# Estimated params after normalizing by repeatings [0.05743187 3.16078721]
# clipbucket_20p_to_30p
# Estimated params after normalizing by repeatings [0.04756999 3.53742806]
# top30p_rand10p
# Estimated params after normalizing by repeatings [0.05165699 1.         3.49290572]
# Estimated params after normalizing by repeatings [0.04756999 3.53742806]
# clipbucket_30p_to_40p
# Estimated params after normalizing by repeatings [0.03595843 3.20321982]
# clipbucket_40p_to_50p
# Estimated params after normalizing by repeatings [0.02315386 3.25426389]
# clipbucket_50p_to_60p
# Estimated params after normalizing by repeatings [0.01451045 2.75617584]
# whole data random sampling
# Estimated params after normalizing by repeatings [0.0250887  3.25373911]

import numpy as np
import matplotlib.pyplot as plt

# params_list = [ [0.02400279 , 1.        ], [0.04672413 , 3.74046299], [0.05743187 , 3.16078721], [0.04756999 , 3.53742806], [0.03595843 , 3.20321982], [0.02315386 , 3.25426389], [0.01451045,  2.75617584], [0.0250887 , 3.25373911] ]
params_list = [[0.04672413 , 3.74046299], [0.05743187 , 3.16078721], [0.04756999 , 3.53742806], [0.05165699 , 3.49290572], [0.03595843 , 3.20321982], [0.02315386 , 3.25426389], [0.01451045,  2.75617584] ]


def a_at_every_epoch(a, half_life, max_epochs=100):
    base = 0.5
    a_list = []
    for i in range(max_epochs):
        a_ = a * base**(i/half_life)
        a_list.append(a_)
    return a_list

def samples_seen_factor(max_epochs=100):
    data_util_list = []
    b = 0.9
    for i in range(max_epochs):
        data_util = (i+1)**b - i**b
        data_util_list.append(data_util)
    return data_util_list

def samples_seen_factor_k(k):
    b = 0.9
    data_util = (k)**b - (np.floor(k-0.1))**b
    return data_util

#utility[k] = a_at_ever_epoch[k] * samples_seen_factor[k]
all_utilities = []
for param in params_list:
    a = param[0]
    half_life = param[1]
    a_at_ever_epoch = a_at_every_epoch(a, half_life, max_epochs=100)
    samples_seen_factor_ = samples_seen_factor(max_epochs=100)
    # utility = [a_at_ever_epoch[k] * samples_seen_factor_[k] for k in range(100)]
    # utility = [a_at_ever_epoch[k]*samples_seen_factor_[k] for k in range(100)]
    utility = [a_at_ever_epoch[k] for k in range(100)]
    #add this list to a list of lists
    # all_utilities = all_utilities + utility
    all_utilities.append(utility)

l1 = [x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[0][:5])]
l2 = [x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[1][:5])]
l3 = [x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[2][:5])]
print(l1)
print(l2)
print(l3)
l_real = [x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[3][:5])]
print("l_real")
print(l_real)

l_avg = [(l1[i] + l2[i] + l3[i])/3 for i in range(5)]
print("l_avg")
print(l_avg)

err = [abs(l_avg[i] - l_real[i])/l_real[i] for i in range(5)]
print("err")
print(err)

change_l1_l2 = [abs(l1[i] - l2[i])/l1[i] for i in range(5)]
print("change_l1_l2")
print(change_l1_l2)

change_l1_l3 = [abs(l1[i] - l3[i])/l1[i] for i in range(5)]
print("change_l1_l3")
print(change_l1_l3)

change_l2_l3 = [abs(l2[i] - l3[i])/l2[i] for i in range(5)]
print("change_l2_l3")
print(change_l2_l3)

# p11 = sum([x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[0][:5])])
# p12 = sum([x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[0][:2])]) + sum([x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[1][:2])]) + sum([x*samples_seen_factor_k(2.5) for i,x in enumerate(all_utilities[0][2:3])])/2 + sum([x*samples_seen_factor_k(2.5) for i,x in enumerate(all_utilities[1][2:3])])/2
# p13 = sum([x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[0][:1])]) + sum([x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[1][:1])]) + sum([x*samples_seen_factor_k(i+1) for i,x in enumerate(all_utilities[2][:1])]) + (sum([x*samples_seen_factor_k(1.66) for i,x in enumerate(all_utilities[0][1:2])]) + sum([x*samples_seen_factor_k(1.66) for i,x in enumerate(all_utilities[1][1:2])]) + sum([x*samples_seen_factor_k(1.66) for i,x in enumerate(all_utilities[2][1:2])]))/3


# p2 = sum(all_utilities[0][:2]) + sum(all_utilities[1][:2]) + 
# epochs = 5
# for i in range(epochs):
#     j = epochs - i
#     #sum the score of first j elements from ith list

# #get the index of the top 10 elemets of the list (in ordere of decreasing utility)
# top_10_indices = np.argsort(all_utilities)[-50:][::-1]

# top_10_indices = [(x//100)/10 + 0.1 for x in top_10_indices]

# #now sum the scores of indices from same list
# #make a dict with vkeys as number o to 9 and values as 0
# # list_scores = {}
# # for i in range(10):
# #     list_scores[i] = 0
# # for index in top_10_indices:
    
# #     list_number = index//100
# #     list_scores[list_number] += all_utilities[index]
# #print the dict
# # print(list_scores)
# print(top_10_indices)

# # print each element of list space separeted with just one decimal place
# for index in top_10_indices:
#     print("{:.1f}".format(index), end=" ")
# print("\n")


# import pdb; pdb.set_trace()


