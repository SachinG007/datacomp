import json
curriculum = [0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.4, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.2, 0.4, 0.5, 0.1, 0.3, 0.2, 0.4, 0.5, 0.1, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.6, 0.1, 0.2, 0.5, 0.4, 0.3, 0.6, 0.1, 0.2, 0.3, 0.5, 0.4, 0.1, 0.6, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.3]

#calculate frequency of each skill
def skill_freq(curriculum):
    skill_freq = {}
    for skill in curriculum:
        if skill in skill_freq:
            skill_freq[skill] += 1
        else:
            skill_freq[skill] = 1
    return skill_freq

skill_freq = skill_freq(curriculum)

#order the dictionary by value
def order_dict(skill_freq):
    skill_freq = dict(sorted(skill_freq.items(), key=lambda item: item[1]))
    return skill_freq
skill_freq = order_dict(skill_freq)
print(skill_freq)

#create datapath as "datacomp_L14_${start}_to_${end}" where start=key=0.1 and end=0.1
def create_datapath(skill_freq):
    datapath = []
    for key in skill_freq:
        key_1 = key - 0.1
        key_1 = round(key_1, 1)
        datapath.append("datacomp_L14_" + str(key_1) + "_to_" + str(key))
    return datapath

datapaths = create_datapath(skill_freq)

def merge_paths(datapaths):
    merged_paths = ""
    for i in range(0, len(datapaths)):
        merged_paths+=datapaths[i] + "::"
    return merged_paths[:-2]

#enumerate the skills dict
path_epochs_dict = {}
earlier_eps = 0
for i, key in enumerate(skill_freq):
    datapath = merge_paths(datapaths[i:])
    epochs_to_do = (skill_freq[key]-earlier_eps)*len(datapaths[i:])
    if epochs_to_do>0:
        path_epochs_dict[datapath]=epochs_to_do
    earlier_eps = skill_freq[key]
print(path_epochs_dict)
for key in path_epochs_dict:
    print(key,"\t")
#save the dict
with open('path_epochs_dict.json', 'w') as fp:
    json.dump(path_epochs_dict, fp)

# datacomp_L14_0.5_to_0.6::datacomp_L14_0.4_to_0.5::datacomp_L14_0.3_to_0.4::datacomp_L14_0.1_to_0.2::datacomp_L14_0.2_to_0.3::datacomp_L14_0.0_to_0.1 datacomp_L14_0.4_to_0.5::datacomp_L14_0.3_to_0.4::datacomp_L14_0.1_to_0.2::datacomp_L14_0.2_to_0.3::datacomp_L14_0.0_to_0.1 datacomp_L14_0.3_to_0.4::datacomp_L14_0.1_to_0.2::datacomp_L14_0.2_to_0.3::datacomp_L14_0.0_to_0.1 datacomp_L14_0.1_to_0.2::datacomp_L14_0.2_to_0.3::datacomp_L14_0.0_to_0.1 datacomp_L14_0.2_to_0.3::datacomp_L14_0.0_to_0.1

    
