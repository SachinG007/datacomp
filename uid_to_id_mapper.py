import numpy as np
import os
# first load all_uids_npy

root = "/project_data/datasets/datanet/metadata/"

sorted_indices_for_uids_in_lexicographic_order =  np.load(os.path.join(root, "sorted_indices_for_uids_in_lexicographic_order.npy"))
sorted_uids_in_lexicographic_order = np.load(os.path.join(root, "sorted_uids_in_lexicographic_order.npy"))

import sys
uid_path = sys.argv[1]

uids = np.load(uid_path)
# hex = format(uid[0], '016x') + format(uid[1], '016x')
hex_uids = np.array([format(uid[0], '016x') + format(uid[1], '016x') for uid in uids])
# this must be a list of uids in lexicographic sorted order

# sort it in lexicographic order if it is not already
hex_uids.sort()


#make is valid of length sorted_indices_for_uids_in_lexicographic_order of 0s
#is_valid = np.zeros(sorted_indices_for_uids_in_lexicographic_order.shape[0])
is_valid = np.zeros(130000000)

current_pointer_in_sorted_uids_in_lexicographic_order = 0
for hex in hex_uids:
    #print(hex)
    # find the index of this hex in sorted_uids_in_lexicographic_order
    # increase the current_pointer_in_sorted_uids_in_lexicographic_order until you find the hex
    while sorted_uids_in_lexicographic_order[current_pointer_in_sorted_uids_in_lexicographic_order] != hex:
        #print("waiting ", current_pointer_in_sorted_uids_in_lexicographic_order)
        current_pointer_in_sorted_uids_in_lexicographic_order += 1
    # now you have the index
    true_index = sorted_indices_for_uids_in_lexicographic_order[current_pointer_in_sorted_uids_in_lexicographic_order]

    is_valid[true_index] = 1
    current_pointer_in_sorted_uids_in_lexicographic_order += 1
    print("Current pointer is " + str(current_pointer_in_sorted_uids_in_lexicographic_order))

# save is_valid at same path as uids but with _is_valid appended
# np.save(uid_path[:-4] + "_is_valid.npy", is_valid)

#save as torch tensor and make bool to save space
import torch
all_uids_torch = torch.from_numpy(is_valid)
all_uids_torch = all_uids_torch.bool()
# save at same place with .pt extension
torch.save(all_uids_torch, uid_path[:-4] + "_is_valid.pt")
