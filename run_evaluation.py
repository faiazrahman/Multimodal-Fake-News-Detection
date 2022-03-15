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
from model import JointTextImageModel, JointTextImageDialogueModel, MultimodalFakeNewsDetectionModel, MultimodalFakeNewsDetectionModelWithDialogue, PrintCallback

# Multiprocessing for dataset batching: NUM_CPUS=24 on Yale Tangra server
# Set to 0 and comment out torch.multiprocessing line if multiprocessing gives errors
NUM_CPUS = 0
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
TRAIN_DATA_SIZE = 100 # TODO 10000
TEST_DATA_SIZE = 1000
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0, 1]

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

def get_checkpoint_filename_from_dir(path):
    return os.listdir(path)[0]

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
    args.trained_model_version = config.get("trained_model_version", None)
    args.trained_model_path = config.get("trained_model_path", None)
    args.preprocessed_train_dataframe_path = config.get("preprocessed_train_dataframe_path", None)
    args.preprocessed_test_dataframe_path = config.get("preprocessed_test_dataframe_path", None)

    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        image_transform = JointTextImageDialogueModel.build_image_transform()
    else:
        image_transform = JointTextImageModel.build_image_transform()

    test_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_test_dataframe_path,
        data_path=args.test_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        images_dir=IMAGES_DIR,
        num_classes=args.num_classes
    )
    logging.info("Test dataset size: {}".format(len(test_dataset)))
    logging.info(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_CPUS
    )
    logging.info(test_loader)

    hparams = {
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": args.num_classes
    }

    checkpoint_path = None
    if args.trained_model_version:
        assets_version = None
        if isinstance(args.trained_model_version, int):
            assets_version = "version_" + str(args.trained_model_version)
        elif isinstance(args.trained_model_version, str):
            assets_version = args.trained_model_version
        else:
            raise Exception("assets_version must be either an int (i.e. the version number, e.g. 16) or a str (e.g. \"version_16\"")
        checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    elif args.trained_model_path:
        checkpoint_path = args.trained_model_path
    else:
        raise Exception("A trained model must be specified for evaluation, either by version number (in default PyTorch Lightning assets path ./lightning_logs) or by custom path")

    checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    logging.info(checkpoint_path)

    model = None
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        model = MultimodalFakeNewsDetectionModelWithDialogue.load_from_checkpoint(checkpoint_path)
    else:
        model = MultimodalFakeNewsDetectionModel.load_from_checkpoint(checkpoint_path)

    trainer = None
    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        callbacks = [PrintCallback()]
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer()
    logging.info(trainer)

    trainer.test(model, dataloaders=test_loader)
    # pl.LightningModule has some issues displaying the results automatically
    # As a workaround, we can store the result logs as an attribute of the
    # class instance and display them manually at the end of testing
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    results = model.test_results

    print(args.test_data_path)
    print(checkpoint_path)
    print(results)
    logging.info(args.test_data_path)
    logging.info(checkpoint_path)
    logging.info(results)
