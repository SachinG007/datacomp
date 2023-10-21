root = "/scratch2/shards/"
import os, tarfile, json
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures


#load all parquet files one by one. find the key "uid"

all_files = os.listdir(root)
all_parquet_files = [file for file in all_files if file[-8:] == ".parquet"]
all_tar_files = [file for file in all_files if file[-4:] == ".tar"]

all_uids_old = np.load("/project_data/datasets/datanet/metadata/all_uids.npy")

#sort
all_parquet_files.sort()
all_tar_files.sort()

null_str = "0"*32
null_int = int(null_str, 16)
all_uids_processed = np.array([(null_int, null_int)]* 130_000_000, np.dtype("u8,u8"))
all_uids = np.array([null_str] *130_000_000)


def process_file(file):
    try:
        df = pd.read_parquet(os.path.join(root, file))
        uids = list(df["uid"])
        keys = list(df["key"])
        keys = [int(key) for key in keys]
        processed_uids = [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids]
        all_uids_processed[keys] = processed_uids
        all_uids[keys] = uids
    except:
        print(file)
        tarfile_path = file[:-8] + ".tar"
        process_file_tar(tarfile_path)

def process_file_tar(tar_file_path):
    with tarfile.open(os.path.join(root, tar_file_path), 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.json'):
                file = tar.extractfile(member)
                try:
                    data = json.load(file)
                    uid_value = (data['uid'])
                    key_value = int(data['key'])
                    uid = (int(uid_value[:16], 16), int(uid_value[16:32], 16))
                    all_uids_processed[key_value] = uid
                    all_uids[key_value] = uid_value

                except json.JSONDecodeError:
                    print(f"Failed to parse {member.name} as JSON")

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(process_file, all_parquet_files), total=len(all_parquet_files)))
        # list(tqdm(executor.map(process_file_tar, all_tar_files), total=len(all_tar_files)))
    # uid = all_uids_old[int(keys[-1])]
    # hex = format(uid[0], '016x') + format(uid[1], '016x')

          
np.save(os.path.join("/scratch2/", "all_uids_npy.npy"), all_uids)
np.save(os.path.join("/scratch2/", "all_uids_processed_npy.npy"), all_uids_processed)

# #sort the uid array in lexico order and create a sorted_indices array
sorted_indices = np.argsort(all_uids, axis=0)
sorted_uids_processed = all_uids_processed[sorted_indices]

# #save all three files

np.save(os.path.join("/scratch2/", "sorted_indices_for_uids_in_lexicographic_order.npy"), sorted_indices)
np.save(os.path.join("/scratch2/", "sorted_uids_in_lexicographic_order.npy"), sorted_uids_processed)
