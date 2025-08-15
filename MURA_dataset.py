# References:
# Adapted from ViTShapley:
# I. Covert, C. Kim, and S.-I. Lee, "Learning to estimate Shapley values with vision transformers,"
# presented at the International Conference on Learning Representations 2023, Kigali, Rwanda, 2023.

import os
import random
import json
import lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import sklearn.model_selection


def save_json(data, filename):
    filename = os.path.abspath(filename)
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as wfile:
        json.dump(data, wfile)

def load_json(filename):
    filename = os.path.abspath(filename)
    with open(filename, "r") as rfile:
        return json.load(rfile)

def pil_loader(img_path, n_channels=3):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 3:
            return img.convert('RGB')
        elif n_channels == 1:
            return img.convert('L')
        elif n_channels == 4:
            return img.convert('RGBA')
        else:
            raise NotImplementedError(
                "PIL only supports 1, 3, or 4 channel inputs. "
                "Use cv2 for more flexible options."
            )

class MURADataset(Dataset):
    def __init__(self, dataset_location, transform_params=None, img_channels=3, body_part_filter=None, split='train'):
        super().__init__()
        # ImageNet mean/std
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.label_map = {'negative': 0, 'positive': 1}
        self.body_part_strs = [
            'XR_ELBOW',
            'XR_FINGER',
            'XR_FOREARM',
            'XR_HAND',
            'XR_HUMERUS',
            'XR_SHOULDER',
            'XR_WRIST'
        ]
        self.body_part_to_idx = {part: i for i, part in enumerate(self.body_part_strs)}
        self.dataset_location = os.path.abspath(dataset_location)
        self.split = split
        self.img_channels = img_channels
        self.transform_params = transform_params
        self.body_part_filter = body_part_filter  # filter by body part for stage two classifier training
        self.data = self.get_data_list()

    def get_data_list(self):
        if self.split in ['train', 'val']:
            csv_name = 'train_image_paths.csv'
            check_str = 'train'
        elif self.split == 'test':
            csv_name = 'valid_image_paths.csv'
            check_str = 'valid'
        else:
            raise ValueError("split must be 'train', 'val', or 'test', got: {}".format(self.split))

        csv_path = os.path.join(self.dataset_location, csv_name)
        data = pd.read_csv(csv_path, header=None, names=["path"])

        # example path: 'MURA-v1.1/train/XR_ELBOW/patient00001/study1_positive/image1.png'
        data["split_check"] = data["path"].map(lambda x: x.split("/")[1])
        assert (data["split_check"] == check_str).all(), (
            f"MURA dataset init: found mismatch in {check_str} paths."
        )

        data["body_part_str"] = data["path"].map(lambda x: x.split("/")[2])
        assert data["body_part_str"].str.startswith("XR").all(), (
            "MURA dataset init: body_part does not start with 'XR'."
        )

        data["patient"] = data["path"].map(lambda x: x.split("/")[3])
        assert data["patient"].str.startswith("patient").all(), (
            "MURA dataset init: patient folder does not start with 'patient'."
        )

        data["study"] = data["path"].map(lambda x: x.split("/")[4].split("_")[0])
        assert data["study"].str.startswith("study").all(), (
            "MURA dataset init: study name does not start with 'study'."
        )

        data["label_str"] = data["path"].map(lambda x: x.split("/")[4].split("_")[1])
        assert data["label_str"].isin(["positive", "negative"]).all(), (
            "MURA dataset init: label not in ['positive', 'negative']."
        )

        if self.split in ['train', 'val']:
            patients_unique = data["patient"].unique()
            idx_train, idx_val = sklearn.model_selection.train_test_split(
                patients_unique, random_state=42, test_size=0.1
            )
            if self.split == 'train':
                data = data[data["patient"].isin(idx_train)]
            else:  # self.split == 'val'
                data = data[data["patient"].isin(idx_val)]

        if self.body_part_filter is not None:
            data = data[data["body_part_str"] == self.body_part_filter]

        # convert paths to absolute paths and map labels
        data_list = []
        for _, row in data.iterrows():
            # remove the first path component (e.g., "MURA-v1.1/train/...")
            rel = row["path"].split("/", 1)[1]
            abs_path = os.path.join(self.dataset_location, rel)
            data_list.append({
                "img_path": abs_path,
                "body_part_idx": self.body_part_to_idx[row["body_part_str"]],
                "body_part_str": row["body_part_str"],
                "patient": row["patient"],
                "study": row["study"],
                "label": self.label_map[row["label_str"]],
            })
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_path = record["img_path"]
        path_png = os.path.splitext(img_path)[0] + '.png'
        if os.path.exists(path_png):
            img = pil_loader(path_png, self.img_channels)
        else:
            img = pil_loader(img_path, self.img_channels)
        if self.transform_params is not None:
            img = self.transform_params(img)
        return {
            "img_path": img_path,
            "body_part_idx": F.one_hot(torch.tensor(record["body_part_idx"]), num_classes=len(self.body_part_strs)).float(),
            "body_part_str": record["body_part_str"],
            "patient": record["patient"],
            "study": record["study"],
            "label": torch.tensor(record["label"]),
            "images": img,
        }

    @staticmethod
    def get_validation_ids(total_size=None, val_size=None, json_path=None, seed_n=42):
        if (total_size is not None and val_size is not None) and json_path is not None:
            if val_size < 1:
                val_size = int(total_size * val_size)
            total_ids = list(range(total_size))
            random.Random(seed_n).shuffle(total_ids)
            train_split = total_ids[val_size:]
            val_split = total_ids[:val_size]
            json_data = {"train_split": train_split, "val_split": val_split}
            save_json(json_data, json_path)
            return train_split, val_split
        elif json_path is not None and (total_size is None and val_size is None):
            json_data = load_json(json_path)
            return json_data["train_split"], json_data["val_split"]
        else:
            raise ValueError("Invalid arguments for 'get_validation_ids'.")