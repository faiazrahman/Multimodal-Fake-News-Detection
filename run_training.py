import os
import logging
import argparse

# pylint: disable=import-error
from tqdm import tqdm # pylint: disable=unused-import
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from sentence_transformers import SentenceTransformer

from dataloader import MultimodalDataset, Modality
from model import JointTextImageModel, JointTextImageDialogueModel, \
    MultimodalFakeNewsDetectionModel, MultimodalFakeNewsDetectionModelWithDialogue, \
    PrintCallback

# Multiprocessing for dataset batching: NUM_CPUS=24 on Yale Tangra server
# Set to 0 and comment out torch.multiprocessing line if multiprocessing gives errors
NUM_CPUS = 0
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0, 1]

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Running on training data")
    parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    # We default all hyperparameters to None so that their default values can
    # be taken from a config file; if the config file is not specified, then we
    # use the given default values in the `config.get()` calls (see below)
    # Thus the order of precedence for hyperparameter values is
    #   passed manually as an arg -> specified in given config file -> default
    # This allows experiments defined in config files to be easily replicated
    # while tuning specific parameters via command-line args
    parser.add_argument("--modality", type=str, default=None, help="text | image | text-image | text-image-dialogue")
    parser.add_argument("--num_classes", type=int, default=None, help="2 | 3 | 6")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--text_embedder", type=str, default=None, help="all-mpnet-base-v2 | all-distilroberta-v1")
    parser.add_argument("--dialogue_summarization_model", type=str, default=None, help="None=Transformers.Pipeline default i.e. sshleifer/distilbart-cnn-12-6 | bart-large-cnn | t5-small | t5-base | t5-large")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_train_dataframe_path", type=str, default=None)
    parser.add_argument("--preprocessed_test_dataframe_path", type=str, default=None)
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.load(yaml_file)

    # pylint: disable=multiple-statements
    if not args.modality: args.modality = config.get("modality", "text-image")
    if not args.num_classes: args.num_classes = config.get("num_classes", 2)
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 10)
    if not args.dropout_p: args.dropout_p = config.get("dropout_p", 0.1)
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)
    if not args.text_embedder:
        args.text_embedder = config.get("text_embedder", "all-mpnet-base-v2")
    if not args.dialogue_summarization_model:
        args.dialogue_summarization_model = config.get("dialogue_summarization_model", "bart-large-cnn")
    if not args.train_data_path:
        args.train_data_path = config.get("train_data_path", os.path.join(DATA_PATH, "multimodal_train_" + str(TRAIN_DATA_SIZE) + ".tsv"))
    if not args.test_data_path:
        args.test_data_path = config.get("test_data_path", os.path.join(DATA_PATH, "multimodal_test_" + str(TEST_DATA_SIZE) + ".tsv"))
    if not args.preprocessed_train_dataframe_path:
        args.preprocessed_train_dataframe_path = config.get("preprocessed_train_dataframe_path", None)
    if not args.preprocessed_test_dataframe_path:
        args.preprocessed_test_dataframe_path = config.get("preprocessed_test_dataframe_path", None)

    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None # pylint: disable=invalid-name
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        image_transform = JointTextImageDialogueModel.build_image_transform()
    else:
        image_transform = JointTextImageModel.build_image_transform()

    train_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_train_dataframe_path,
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
    callbacks = [PrintCallback()]
    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks
        )
    logging.info(trainer)

    trainer.fit(model, train_loader)
