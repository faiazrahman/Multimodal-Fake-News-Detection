import sys
import os
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
# from torchvision import transforms

import pytorch_lightning as pl
# from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sentence_transformers import SentenceTransformer

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"
RESNET_OUT_DIM = 2048

logging.basicConfig(level=logging.DEBUG) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())

class MultimodalDataset(Dataset):

    def __init__(self, data_path, image_transform, num_classes=2, images_dir=IMAGES_DIR):
        df = pd.read_csv(data_path, sep='\t', header=0)
        df = self._preprocess_df(df)
        print(df.columns)
        print(df['clean_title'])
        self.data_frame = df
        logging.debug(self.data_frame)

        self.label = "2_way_label"
        if num_classes == 3:
            self.label = "3_way_label"
        elif num_classes == 6:
            self.label = "6_way_label"

        self.image_transform = image_transform

        return

    def __len__(self):
        return len(self.data_frame.index)

    def __getitem__(self, idx):
        """ Currently returning text string and image RGB Tensor """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_id = self.data_frame.loc[idx, 'id']
        image_path = os.path.join(IMAGES_DIR, item_id + IMAGE_EXTENSION)
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        text = self.data_frame.loc[idx, 'clean_title']

        label = torch.Tensor(
            [self.data_frame.loc[idx, self.label]]
        ).long().squeeze()

        item = {
            "id": item_id,
            "text": text,
            "image": image,
            "label": label
        }

        return item

    @staticmethod
    def _preprocess_df(df):
        def image_exists(row):
            image_path = os.path.join(IMAGES_DIR, row['id'] + IMAGE_EXTENSION)
            return os.path.exists(image_path)

        df['image_exists'] = df.apply(lambda row: image_exists(row), axis=1)
        df = df[df['image_exists'] == True].drop('image_exists', axis=1)
        df = df.drop(['created_utc', 'domain', 'hasImage', 'image_url'], axis=1)
        df.reset_index(drop=True, inplace=True)
        return df

class JointVisualTextualModel(nn.Module):

    def __init__(
            self,
            num_classes,
            text_module,
            image_module,
            text_feature_dim,
            image_feature_dim,
            fusion_output_size
        ):
        super(JointVisualTextualModel, self).__init__()
        pass

    def forward(self, text, image, label):
        pass

class MultimodalFakeNewsDetectionModel(pl.LightningModule):

    def __init__(self, hparams):
        super(MultimodalFakeNewsDetectionModel, self).__init__()
        self.hparams.update(hparams) # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525

        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)

        self.model = self._build_model()

    # Required for pl.LightningModule
    def forward(self, text, image, label):
        return self.model(text, image, label)

    def training_step(self, batch, batch_idx):
        print(batch["text"])
        print(batch["image"])
        print(batch["label"])

        # TODO Get text embeddings before passing to model

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _build_model(self):
        text_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.text_feature_dim)

        image_module = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        image_module.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=self.image_feature_dim)

        return JointVisualTextualModel(
            num_classes=self.hparams.get("num_classes", 2),
            text_module=text_module,
            image_module=image_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512)
        )


def _build_text_transform():
    pass

def _build_image_transform(image_dim=224):
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(image_dim, image_dim)),
        torchvision.transforms.ToTensor(),
        # All torchvision models expect the same normalization mean and std
        # https://pytorch.org/docs/stable/torchvision/models.html
        torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    return image_transform

if __name__ == "__main__":
    train_data_path = os.path.join(DATA_PATH, "multimodal_train_100.tsv")

    print("Using data:", train_data_path)
    image_transform = _build_image_transform()
    train_dataset = MultimodalDataset(
        train_data_path, image_transform, images_dir=IMAGES_DIR)

    print(len(train_dataset))
    # print(train_dataset[0])

    hparams = {}
    model = MultimodalFakeNewsDetectionModel(hparams)
    model.fit()