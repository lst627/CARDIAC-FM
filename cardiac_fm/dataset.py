import os, sys
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import random
import torchvision.transforms.functional as tf

from fairseq_signals.fairseq_signals.data.ecg.raw_ecg_dataset import FileECGDataset

def transform_train_per_channel(image):
    angle = transforms.RandomRotation.get_params([-90, 90])
    flip = False
    cont = False
    negative = False

    # Random flags
    if random.random() > 0.5:
        flip = True
    if random.random() > 0.5:
        cont = True
    if random.random() > 0.5:
        negative = True

    # Random delta for brightness shift
    delta = torch.rand(1)
    if negative:
        delta = -delta
    delta = delta / 10  # range: [-0.1, 0.1]

    T, C, H, W = image.shape
    for t in range(T):
        for c in range(C):
            # Extract single channel image (H, W)
            img = Image.fromarray(np.uint8(image[t, c, :, :]))

            # Apply random rotation
            img = tf.rotate(img, angle)

            # Optional contrast adjustment
            if cont:
                img = tf.adjust_contrast(img, 2)

            # Optional horizontal flip (commented out, consistent with original)
            # if flip:
            #     img = tf.hflip(img)

            # Convert to tensor and apply delta shift
            img_tensor = tf.to_tensor(img)  # shape: [1, H, W]
            image[t, c, :, :] = img_tensor.squeeze(0) + delta

    return image

def transform_test_per_channel(image):
    T, C, H, W = image.shape
    for t in range(T):
        for c in range(C):
            single_channel = image[t, c, :, :]
            img = Image.fromarray(np.uint8(single_channel))
            img_tensor = tf.to_tensor(img)
            image[t, c, :, :] = img_tensor.squeeze(0)
    return image


class mydataset(Dataset):
    def __init__(self, data_dir, datapath, weighted_sampling=False, split="train", skipped_indices = None):
        df = pd.read_csv(datapath)
        df = df[~df["eid_visit"].isin(skipped_indices)]
        n = len(df) 

        print(f"Total {split} data:", n)
        print(f"Loading {split} data ...")
        
        self.data = []
        self.labels = [0, 1]
        self.class_counts = [0, 0]
        self.weighted_sampling = weighted_sampling

        root_dir = data_dir 
        for i in tqdm(range(n)):
            eid_visit = df.iloc[i]['eid_visit']
            
            mri_label = 1 
            frame_dir = osp.join(root_dir, eid_visit)
            sa_path  = os.path.join(frame_dir, 'cropped_sa_64_normalized2.npy')
            ch2_path = os.path.join(frame_dir, 'cropped_2ch_64_normalized2.npy')
            ch4_path = os.path.join(frame_dir, 'cropped_4ch_64_normalized2.npy')

            # If any of the three required files is missing, skip this sample
            if not (osp.exists(sa_path) and osp.exists(ch2_path) and osp.exists(ch4_path)):
                raise FileNotFoundError(f"Missing required file(s) in {frame_dir}, this sample will be None.")

            modalities =[]
            data_ori = np.load(os.path.join(frame_dir, 'cropped_sa_64_normalized2.npy'))
            slices_indices = [0, 1, 2, 3]
            if data_ori.shape[0]<4:
                raise ValueError(f"data_ori.shape[0]<4 in {frame_dir}, this sample will be None.")
            data_ori = data_ori[slices_indices, ::2, :, :].astype(np.float32)
            modalities.append(data_ori)

            
            data_ori_2ch = np.load(os.path.join(frame_dir, 'cropped_2ch_64_normalized2.npy'))
            data_ori_2ch = data_ori_2ch[:, ::2, :, :].astype(np.float32)
            modalities.append(data_ori_2ch)

            
            data_ori_4ch = np.load(os.path.join(frame_dir,'cropped_4ch_64_normalized2.npy')) 
            data_ori_4ch = data_ori_4ch[:, ::2, :, :].astype(np.float32)
            modalities.append(data_ori_4ch)
        
            #data = data_ori
            data = np.concatenate(modalities, axis=0)
            
            # num_channels, num_frames, width, height
            volume = np.moveaxis(data, 0, 1)
            # num_frames, num_channels, width, height
            self.data.append((volume, mri_label))

        self.map = transform_train_per_channel if split == "train" else transform_test_per_channel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data[index][0] is None:
            return None, self.data[index][1] 
        
        volume, y = self.data[index]
        if self.map:
            x = self.map(volume)
        else:
            x = volume
        return x, y

    def normalize(self, data):
        """
        Normalizes all images in a batch.
        Input: (C, T, H, W)
        Output: (C, T, H, W)
        """
        # Compute per-channel mean and std from the entire batch
        mean = np.mean(data, axis=(0, 2, 3), keepdims=True)  # Shape: (1, T, 1, 1)
        std = np.std(data, axis=(0, 2, 3), keepdims=True) + 1e-6  # Prevent division by zero

        return (data - mean) / std

def crop_to_max_size(sample, target_size, rand=False):
    dim = sample.dim()
    size = sample.shape[-1]
    diff = size - target_size
    if diff <= 0:
        return sample

    start = 0
    if rand:
        start = np.random.randint(0, diff + 1)
    end = size - diff + start
    if dim == 1:
        return sample[start:end], start, end
    elif dim == 2:
        return sample[:, start:end], start, end
    else:
        raise AssertionError('Check the dimension of the input data')


    def normalize(self, data):
        """
        Normalizes all images in a batch.
        Input: (C, T, H, W)
        Output: (C, T, H, W)
        """
        # Compute per-channel mean and std from the entire batch
        mean = np.mean(data, axis=(0, 2, 3), keepdims=True)  # Shape: (1, T, 1, 1)
        std = np.std(data, axis=(0, 2, 3), keepdims=True) + 1e-6  # Prevent division by zero

        return (data - mean) / std

class ECGMRIDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, ecg_dir, mri_dir, labels_dir=None, mask=None, split="train"):
        """
        Custom Dataset for loading ECG and MRI data.

        Parameters:
            csv_path (str): Path to the master CSV file
            ecg_dir (str): Directory containing ECG data files.
            mri_dir (str): Directory containing MRI data files.
            mask (np.ndarray or None): Optional mask array. If None, a default mask with the same shape as the ECG data is created, filled with False.
        """

        self.ecg_dir = ecg_dir
        self.mri_dir = mri_dir
        self.mask = mask
        self.label = labels_dir != None
        # The hyperparameter is from fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer/physionet_finetuned.yaml
        
        manifest_path = os.path.join(ecg_dir, "{}.tsv".format(split)) 

        self.ecg_data = FileECGDataset(
                manifest_path=manifest_path,
                sample_rate=None,
                max_sample_size=None,
                min_sample_size=None,
                pad=True,
                pad_leads=False,
                leads_to_load=None,
                label=self.label, # True
                label_file=labels_dir,
                filter=False,
                normalize=False,
                mean_path=None,
                std_path=None,
                num_buckets=0,
                compute_mask_indices=False,
                leads_bucket=None,
                bucket_selection="uniform",
                training=True if 'train' in split else False,
            )
        ## NOTE: We changed this line for 3+3!
        self.mri_data = mydataset( 
                data_dir=mri_dir, 
                datapath=csv_path, 
                split=split,
                skipped_indices=self.ecg_data.skipped_indices
        )

    def __len__(self):
        return len(self.mri_data)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]
        mri = self.mri_data[idx][0]
        mri = torch.tensor(mri, dtype=torch.float32)
        return mri, ecg

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle batches of ECG and MRI data.

        Parameters:
            batch (list): List of tuples (MRI, ECG).

        Returns:
            tuple: Collated MRI and ECG data.
        """
        ecg_batch = [item[1] for item in batch]
        mri_batch = [item[0] for item in batch]

        # Use FileECGDataset's collator for ECG data

        sources = [s["source"] for s in ecg_batch]
        if "label" in ecg_batch[0]:
            labels = [s["label"] for s in ecg_batch]
        else: labels = None

        originals = None
        sizes = [s.size(-1) for s in sources]

        target_size = min(max(sizes), sys.maxsize) # self.pad is always True
        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        collated_originals = collated_sources.clone() if originals else None
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) # if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                if originals:
                    collated_originals[i] = originals[i]
            elif diff < 0:
                # assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                if originals:
                    collated_originals[i] = torch.cat(
                        [originals[i], originals[i].new_full((originals[i].shape[0], -diff,), 0.0)], dim=-1
                    )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i], start, end = crop_to_max_size(source, target_size, rand=True)
                if originals:
                    collated_originals[i] = originals[i][:,start:end]

        input = {"source": collated_sources}
        ecg_collated = {"id": torch.LongTensor([s["id"] for s in ecg_batch])}
        # if self.label:
        #     ecg_collated["label"] = torch.stack([s["label"] for s in samples])

        if originals:
            ecg_collated["original"] = collated_originals

        # if self.pad:
        input["padding_mask"] = padding_mask

        ecg_collated["net_input"] = input

        # Collate MRI data
        mri_collated = torch.stack(mri_batch)

        if labels:
            ecg_collated["label"] = torch.stack(labels)

        return mri_collated, ecg_collated
    

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, ecg_dir, labels_dir, mask=None, split="train"):
        """
        Custom Dataset for loading ECG data.

        Parameters:
            csv_path (str): Path to the master CSV file
            ecg_dir (str): Directory containing ECG data files.
            mask (np.ndarray or None): Optional mask array. If None, a default mask with the same shape as the ECG data is created, filled with False.
        """

        self.ecg_dir = ecg_dir
        self.mask = mask
        # The hyperparameter is from fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer/physionet_finetuned.yaml
        # Not adding perturbation 
        manifest_path = os.path.join(ecg_dir, "{}.tsv".format(split))

        self.ecg_data = FileECGDataset(
                manifest_path=manifest_path,
                sample_rate=None,
                max_sample_size=None,
                min_sample_size=None,
                pad=True,
                pad_leads=False,
                leads_to_load=None,
                label=True,
                label_file=labels_dir,
                filter=False,
                normalize=False,
                mean_path=None,
                std_path=None,
                num_buckets=0,
                compute_mask_indices=False,
                leads_bucket=None,
                bucket_selection="uniform",
                training=True if 'train' in split else False,
            )

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx] 
        return ecg

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle batches of ECG data.

        Parameters:
            batch (list): List of tuples (MRI, ECG).

        Returns:
            tuple: Collated ECG data.
        """
        ecg_batch = batch
        # Use FileECGDataset's collator for ECG data

        sources = [s["source"] for s in ecg_batch]
        labels = [s["label"] for s in ecg_batch]
        
        originals = None
        sizes = [s.size(-1) for s in sources]
        # assert using pad == True
        target_size = min(max(sizes), sys.maxsize) # self.max_sample_size)

        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        collated_originals = collated_sources.clone() if originals else None
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) # if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                if originals:
                    collated_originals[i] = originals[i]
            elif diff < 0:
                # assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                if originals:
                    collated_originals[i] = torch.cat(
                        [originals[i], originals[i].new_full((originals[i].shape[0], -diff,), 0.0)], dim=-1
                    )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i], start, end = crop_to_max_size(source, target_size, rand=True)
                if originals:
                    collated_originals[i] = originals[i][:,start:end]

        input = {"source": collated_sources}
        ecg_collated = {"id": torch.LongTensor([s["id"] for s in ecg_batch])}

        if originals:
            ecg_collated["original"] = collated_originals
            
        input["padding_mask"] = padding_mask
        ecg_collated["net_input"] = input
        
        if labels:
            ecg_collated["label"] = torch.stack(labels)
       
        return ecg_collated


