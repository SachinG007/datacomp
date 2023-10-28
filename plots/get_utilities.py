# mediumscale_nofilter
# Estimated params after normalizing by repeatings [0.02400279 1.        ]
# clipbucket_top10p
# Estimated params after normalizing by repeatings [0.04672413 3.74046299]
# clipbucket_10p_to_20p
# Estimated params after normalizing by repeatings [0.05743187 3.16078721]
# clipbucket_top30p_10prandom
# Estimated params after normalizing by repeatings [0.05165699 3.49290572]
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

params_list = [ [0.02400279 , 1.        ], [0.04672413 , 3.74046299], [0.05743187 , 3.16078721], [0.05165699 , 3.49290572], [0.03595843 , 3.20321982], [0.02315386 , 3.25426389], [0.01451045,  2.75617584], [0.0250887 , 3.25373911] ]


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

#utility[k] = a_at_ever_epoch[k] * samples_seen_factor[k]
all_utilities = []
for param in params_list:
    a = param[0]
    half_life = param[1]
    a_at_ever_epoch = a_at_every_epoch(a, half_life, max_epochs=100)
    samples_seen_factor_ = samples_seen_factor(max_epochs=100)
    utility = [a_at_ever_epoch[k] * samples_seen_factor_[k] for k in range(100)]
    #add this list to a list of lists
    all_utilities = all_utilities + utility


#get the index of the top 10 elemets of the list (in ordere of decreasing utility)
top_10_indices = np.argsort(all_utilities)[-10:]

#now sum the scores of indices from same list
#make a dict with vkeys as number o to 9 and values as 0
list_scores = {}
for i in range(10):
    list_scores[i] = 0
for index in top_10_indices:
    
    list_number = index//100
    list_scores[list_number] += all_utilities[index]
#print the dict
print(list_scores)
import pdb; pdb.set_trace()


