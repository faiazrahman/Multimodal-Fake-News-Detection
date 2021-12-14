import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
import numpy as np

from tqdm import tqdm
import yaml

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl

from sentence_transformers import SentenceTransformer


from dataloader import MultimodalDataset, Modality
from model import JointVisualTextualModel, JointTextImageDialogueModel, MultimodalFakeNewsDetectionModel, MultimodalFakeNewsDetectionModelWithDialogue, PrintCallback

# Multiprocessing for dataset batching: NUM_CPUS=24 on Yale Tangra server
# Set to 0 and comment out torch.multiprocessing line if multiprocessing gives errors
NUM_CPUS = 0
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
TRAIN_DATA_SIZE = 100 # TODO 10000
TEST_DATA_SIZE = 1000
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0, 1]

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Running on training data")
    parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.load(yaml_file)

    args.modality = config.get("modality", "text-image")
    args.num_classes = config.get("num_classes", 2)
    args.batch_size = config.get("batch_size", 32)
    args.learning_rate = config.get("learning_rate", 1e-4)
    args.num_epochs = config.get("num_epochs", 10)
    args.dropout_p = config.get("dropout_p", 0.1)
    args.gpus = config.get("gpus", DEFAULT_GPUS)
    args.text_embedder = config.get("text_embedder", "all-mpnet-base-v2")
    args.dialogue_summarization_model = config.get("dialogue_summarization_model", "bart-large-cnn")
    args.train_data_path = config.get("train_data_path", os.path.join(DATA_PATH, "multimodal_train_" + str(TRAIN_DATA_SIZE) + ".tsv"))
    args.test_data_path = config.get("test_data_path", os.path.join(DATA_PATH, "multimodal_test_" + str(TEST_DATA_SIZE) + ".tsv"))
    args.preprocessed_dataframe_path = config.get("preprocessed_dataframe_path", None)

    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        image_transform = JointTextImageDialogueModel.build_image_transform()
    else:
        image_transform = JointVisualTextualModel.build_image_transform()

    train_dataset = MultimodalDataset(
        from_dialogue_dataframe=args.from_dialogue_dataframe,
        data_path=args.train_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        images_dir=IMAGES_DIR,
        num_classes=args.num_classes
    )
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    logging.info(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_CPUS
    )
    logging.info(train_loader)

    hparams = {
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": args.num_classes
    }

    model = None
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        model = MultimodalFakeNewsDetectionModelWithDialogue(hparams)
    else:
        model = MultimodalFakeNewsDetectionModel(hparams)

    trainer = None
    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        callbacks = [PrintCallback()]
        trainer = pl.Trainer(
            gpus=DEFAULT_GPUS,
            strategy="dp",
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer()
    logging.info(trainer)

    trainer.fit(model, train_loader)
