import torch
from eval_utils.log_reg import LogisticRegression
import numpy as np
import copy
import os
import glob
import collections
from tqdm import tqdm
from torch.utils.data import Dataset

def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch

def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)
    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            image_encoder = image_encoder.to(inputs.device)
            features = image_encoder(inputs)[0]

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data

def get_features(is_train, image_encoder, dataset, dataloader, device, cache_dir, dname):
    split = 'train' if is_train else 'val'
    if cache_dir is not None:
        cache_dir = f'{cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
        
    if cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        data = get_features_helper(image_encoder, dataloader, device)
        if cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, dataloader, device, cache_dir=None, dname="test"):
        self.data = get_features(is_train, image_encoder, dataset, dataloader, device, cache_dir, dname)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def test_log_reg_warm_starting(train_features,
                               test_features,
                               train_labels,
                               test_labels,
                               max_iter=500,
                               random_state=0):
    

    lr_list = [0.1, 0.01, 0.001]
    weight_decay_list = [0, 0.01, 0.0001]
    
    clf_best = None
    best_acc = -1.0
    for lr in lr_list:
        for weight_decay in weight_decay_list:
            clf = LogisticRegression(random_state=random_state,
                                    warm_start=True,
                                    max_iter=max_iter, lr = lr, weight_decay = weight_decay)
            clf.fit(torch.from_numpy(train_features).cuda(), torch.from_numpy(train_labels).cuda())
            test_acc = clf.score(torch.from_numpy(test_features).cuda(), torch.from_numpy(test_labels).cuda())
            print(f"lr: {lr} | wd : {weight_decay} | Test Acc : {test_acc}")
            if test_acc > best_acc:
                best_acc = test_acc
                clf_best = clf

    return clf_best.model.linear

def lbfgs(task, transform, clip_encoder, classification_head, cache_dir = ".", batch_size = 64, num_workers = 4):
    model = clip_encoder

    # Load data
    from eval_utils.wds_eval import create_webdataset
    dataset, dataloader = create_webdataset(
        task, transform, data_root = None, dataset_len = None, batch_size = batch_size, num_workers = 64, split = "train"
    )

    feature_dataset_train = FeatureDataset(is_train=True,
                                     image_encoder=model,
                                     dataset=dataset, dataloader = dataloader,
                                     device=classification_head.device,
                                     cache_dir=cache_dir,
                                     dname = task)

    


    dataset, dataloader = create_webdataset(
        task, transform, data_root = None, dataset_len = None, batch_size = batch_size, num_workers = 4, split = "test"
    )
    
    feature_dataset_val = FeatureDataset(is_train=False,
                                         image_encoder=model,
                                        dataset=dataset, dataloader = dataloader,
                                        device=classification_head.device,
                                        cache_dir=cache_dir,
                                        dname = task)


    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
    classification_head.train()
    model.train()

    train_features, train_labels = feature_dataset_train.data['features'], feature_dataset_train.data['labels']
    test_features, test_labels = feature_dataset_val.data['features'], feature_dataset_val.data['labels']

    head = test_log_reg_warm_starting(
        train_features, test_features, train_labels, test_labels)
    
    if cache_dir is not None:
        os.makedirs(f"{cache_dir}/{task}", exist_ok=True)
        torch.save(head, f"{cache_dir}/{task}/head.pt")
    return head
    