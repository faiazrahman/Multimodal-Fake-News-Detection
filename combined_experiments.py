import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sentence_transformers import SentenceTransformer

import transformers

NUM_CPUS = 0 # 24 on Yale Tangra server; Set to 0 and comment out next line if multiprocessing gives errors
# torch.multiprocessing.set_start_method('spawn')

NUM_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1
MODALITY = "text-image"

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"
RESNET_OUT_DIM = 2048
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768

losses = []

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())

class Modality(enum.Enum):
    """
    Note on Comparisons: Either use `string_value == enum.value`
    or `Modality(string_value) == enum`
    """
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text-image"
    TEXT_IMAGE_DIALOGUE = "text-image-dialogue"

class MultimodalDataset(Dataset):

    def __init__(
        self,
        from_preprocessed_dataframe=None,
        from_dialogue_dataframe=None,
        data_path=None, # Path to data (i.e. not using preprocessed dataframe),
        type="train",
        modality=None,
        text_embedder=None,
        image_transform=None,
        summarization_model=None,
        num_classes=2,
        images_dir=IMAGES_DIR
    ):
        df = None
        if not from_preprocessed_dataframe:
            # This is the first time this data is being setup, so we run full preprocessing
            df = pd.read_csv(data_path, sep='\t', header=0)
            df = self._preprocess_df(df)
            logging.debug(df.columns)
            logging.debug(df['clean_title'])

            # Run dialogue preprocessing, if needed
            if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
                # Special Case: Since the dialogue data is huge, we can first preprocess the
                # dialogue (comments) dataframe to only keep the comments that pertain to
                # posts in our dataset's data, and save it into a serialized .pkl file;
                # If we do that, then we'll run our dialogue preprocessing using that
                # dataframe (which we load from the .pkl file)
                if from_dialogue_dataframe:
                    self._preprocess_dialogue(from_saved_df_path=from_dialogue_dataframe)
                else:
                    self._preprocess_dialogue()
        else: # from_preprocessed_dataframe:
            df = None
            if isinstance(from_preprocessed_dataframe, pd.DataFrame):
                df = from_preprocessed_dataframe
            elif isinstance(from_preprocessed_dataframe, str):
                df = pd.read_pickle(from_preprocessed_dataframe)
            else:
                raise Exception("MultimodalDataset given invalid from_preprocessed_dataframe arg; Must be path (str) to dataframe or pd.DataFrame")

        self.data_frame = df
        self.text_ids = set(self.data_frame['id'])
        logging.debug(self.data_frame)

        self.modality = modality
        self.label = "2_way_label"
        if num_classes == 3:
            self.label = "3_way_label"
        elif num_classes == 6:
            self.label = "6_way_label"

        self.text_embedder = text_embedder
        self.image_transform = image_transform
        self.summarization_model = summarization_model

        self.summarizer = None
        if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE and summarization_model:
            # Model options: "bart-large-cnn", "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
            # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines#transformers.SummarizationPipeline
            self.summarizer = transformers.pipeline("summarization", model=summarization_model)
        elif Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
            self.summarizer = transformers.pipeline("summarization")

        return

    def __len__(self):
        return len(self.data_frame.index)

    def __getitem__(self, idx):
        """
        Returns a text embedding Tensor, image RGB Tensor, and label Tensor
        For data parallel training, the embedding step must happen in the
        torch.utils.data.Dataset __getitem__() method; otherwise, any data that
        is not embedded will not be distributed across the multiple GPUs

        item = {
            "id": item_id,
            "text": text,
            "image": image,
            "dialogue": dialogue,
            "label": label
        }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text, image, dialogue = None, None, None
        item_id = self.data_frame.loc[idx, 'id']

        label = torch.Tensor(
            [self.data_frame.loc[idx, self.label]]
        ).long().squeeze()

        item = {
            "id": item_id,
            "label": label
        }

        if Modality(self.modality) in [Modality.TEXT, Modality.TEXT_IMAGE, Modality.TEXT_IMAGE_DIALOGUE]:
            text = self.data_frame.loc[idx, 'clean_title']
            text = self.text_embedder.encode(text, convert_to_tensor=True)
            item["text"] = text
        if Modality(self.modality) in [Modality.IMAGE, Modality.TEXT_IMAGE, Modality.TEXT_IMAGE_DIALOGUE]:
            image_path = os.path.join(IMAGES_DIR, item_id + IMAGE_EXTENSION)
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)
            item["image"] = image
        if Modality(self.modality) == Modality.TEXT_IMAGE_DIALOGUE:
            dialogue = self.data_frame.loc[idx, 'comment_summary']
            dialogue = self.text_embedder.encode(dialogue, convert_to_tensor=True)
            item["dialogue"] = dialogue

        return item

    @staticmethod
    def _preprocess_df(df):
        def image_exists(row):
            """ Ensures that image exists and can be opened """
            image_path = os.path.join(IMAGES_DIR, row['id'] + IMAGE_EXTENSION)
            if not os.path.exists(image_path):
                return False

            try:
                image = Image.open(image_path)
                image.verify()
                image.close()
                return True
            except Exception:
                return False

        df['image_exists'] = df.apply(lambda row: image_exists(row), axis=1)
        df = df[df['image_exists'] == True].drop('image_exists', axis=1)
        df = df.drop(['created_utc', 'domain', 'hasImage', 'image_url'], axis=1)
        df.reset_index(drop=True, inplace=True)
        return df

    def _preprocess_dialogue(self, from_saved_df_path=""):
        """ A comment's 'submission_id' is linked (i.e. equal) to a post's 'id'
        and 'body' contains the comment text and 'ups' contains upvotes """

        def generate_summaries_and_save_df(df):
            # Add new column in main dataframe to hold dialogue summaries
            self.data_frame['comment_summary'] = ""

            failed_ids = []
            for iteration, text_id in enumerate(self.text_ids):
                if (iteration % 250 == 0):
                    print("Generating summaries for item {}...".format(iteration))
                    # Save progress so far
                    # TODO make this scalable
                    # self.data_frame.to_pickle("./data/text_image_dialogue_dataframe.pkl") # TODO
                    self.data_frame.to_pickle("./data/test__text_image_dialogue_dataframe.pkl") # TODO

                try:
                    # Group comments by post id
                    all_comments = df[df['submission_id'] == text_id]
                    all_comments.sort_values(by=['ups'], ascending=False)
                    all_comments = list(all_comments['body'])

                    # Generate summary of comments via Transformers pipeline
                    corpus = "\n".join(all_comments)
                    summary = "none" # Default if no comments for this post
                    if len(all_comments) > 0:
                        # We define the summary's max_length as max(min(75, num_words // 2), 5)
                        # Note that num_words is calculated very roughly, splitting on whitespace
                        num_words = sum([len(comment.split()) for comment in all_comments])
                        max_length = min(75, num_words // 2) # For short comment threads, it'll be <75
                        max_length = max(max_length, 5) # Avoid 1-length maxes, which leads to unexpected behavior
                        min_length = min(5, max_length - 1)
                        summary = self.summarizer(corpus, min_length=min_length, max_length=max_length, truncation=True)

                        # Pipeline returns a list containing a dict
                        # https://huggingface.co/docs/transformers/master/en/main_classes/pipelines
                        summary = summary[0]['summary_text']

                    # Add summary to self.data_frame 'comment_summary' column
                    self.data_frame.loc[self.data_frame['id'] == text_id, 'comment_summary'] = summary
                except:
                    failed_ids.append(text_id)

            # Dump main df into pkl (and figure out path convention)
            # TODO make this scalable
            # self.data_frame.to_pickle("./data/text_image_dialogue_dataframe.pkl") # TODO
            self.data_frame.to_pickle("./data/test__text_image_dialogue_dataframe.pkl") # TODO

            logging.debug(self.data_frame)
            logging.debug(self.data_frame['comment_summary'])
            logging.debug("num_failed:", len(failed_ids))
            logging.debug(failed_ids)

        if from_saved_df_path != "":
            # Special Case (see above comment in __init__)
            df = pd.read_pickle(from_saved_df_path)
            generate_summaries_and_save_df(df)
        else:
            df = pd.read_csv("./data/all_comments.tsv", sep='\t')
            logging.debug(df)

            def text_exists(row):
                """ Ensures that a comment's corresponding text exists """
                if row['submission_id'] in self.text_ids:
                    return True
                else:
                    return False

            def comment_deleted(row):
                """ If a comment was deleted, its body just contains [deleted] """
                return row['body'] == "[deleted]"

            df['text_exists'] = df.apply(lambda row: text_exists(row), axis=1)
            df = df[df['text_exists'] == True].drop('text_exists', axis=1)
            df['comment_deleted'] = df.apply(lambda row: comment_deleted(row), axis=1)
            df = df[df['comment_deleted'] == False].drop('comment_deleted', axis=1)
            df.reset_index(drop=True, inplace=True)
            logging.debug(df)

            # TODO make this scalable
            # df.to_pickle("./data/comment_dataframe.pkl") # TODO
            df.to_pickle("./data/test__comment_dataframe.pkl") # TODO
            generate_summaries_and_save_df(df)

class JointVisualTextualModel(nn.Module):

    def __init__(
            self,
            num_classes,
            loss_fn,
            text_module,
            image_module,
            text_feature_dim,
            image_feature_dim,
            fusion_output_size,
            dropout_p,
            hidden_size=512,
        ):
        super(JointVisualTextualModel, self).__init__()
        self.text_module = text_module
        self.image_module = image_module
        self.fusion = torch.nn.Linear(in_features=(text_feature_dim + image_feature_dim),
            out_features=fusion_output_size)
        # self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.fc1 = torch.nn.Linear(in_features=fusion_output_size, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, label):
        text_features = torch.nn.functional.relu(self.text_module(text))
        image_features = torch.nn.functional.relu(self.image_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(
            torch.nn.functional.relu(self.fusion(combined))) # TODO add dropout
        # logits = self.fc(fused)
        hidden = torch.nn.functional.relu(self.fc1(fused))
        logits = self.fc2(hidden)

        # nn.CrossEntropyLoss expects raw logits as model output, NOT torch.nn.functional.softmax(logits, dim=1)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        pred = logits
        loss = self.loss_fn(pred, label)

        return (pred, loss)

class MultimodalFakeNewsDetectionModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(MultimodalFakeNewsDetectionModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)

        self.model = self._build_model()

    # Required for pl.LightningModule
    def forward(self, text, image, label):
        # pl.Lightning convention: forward() defines prediction for inference
        return self.model(text, image, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        text, image, label = batch["text"], batch["image"], batch["label"]

        pred, loss = self.model(text, image, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item())
        losses.append(loss.item())
        return loss

    # Optional for pl.LightningModule
    def training_step_end(self, batch_parts):
        """
        Aggregates results when training using a strategy that splits data
        from each batch across GPUs (e.g. data parallel)

        Note that training_step returns a loss, thus batch_parts returns a list
        of 2 loss values (since there are 2 GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        text, image, label = batch["text"], batch["image"], batch["label"]
        pred, loss = self.model(text, image, label)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.sum(pred_label == label).item() / (len(label) * 1.0)
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda()
        }
        print(loss.item(), output['test_acc'])
        return output

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_acc': avg_accuracy
        }

        # pl.LightningModule has some issues displaying the results automatically
        # As a workaround, we can store the result logs as an attribute of the
        # class instance and display them manually at the end of testing
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
        self.test_results = logs

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_accuracy,
            'log': logs,
            'progress_bar': logs
        }

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9)
        return optimizer

    def _build_model(self):
        text_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.text_feature_dim)

        image_module = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        image_module.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=self.image_feature_dim)

        return JointVisualTextualModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )

class JointTextImageDialogueModel(nn.Module):

    def __init__(
            self,
            num_classes,
            loss_fn,
            text_module,
            image_module,
            dialogue_module,
            text_feature_dim,
            image_feature_dim,
            dialogue_feature_dim,
            fusion_output_size,
            dropout_p,
            hidden_size=512,
        ):
        super(JointTextImageDialogueModel, self).__init__()
        self.text_module = text_module
        self.image_module = image_module
        self.dialogue_module = dialogue_module
        self.fusion = torch.nn.Linear(in_features=(text_feature_dim + image_feature_dim + dialogue_feature_dim),
            out_features=fusion_output_size)
        # self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.fc1 = torch.nn.Linear(in_features=fusion_output_size, out_features=hidden_size) # trial
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes) # trial
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, dialogue, label):
        text_features = torch.nn.functional.relu(self.text_module(text))
        image_features = torch.nn.functional.relu(self.image_module(image))
        dialogue_features = torch.nn.functional.relu(self.dialogue_module(dialogue))
        # print(text_features.size(), image_features.size()) # torch.Size([32, 300]) torch.Size([16, 300])
        combined = torch.cat([text_features, image_features, dialogue_features], dim=1)
        fused = self.dropout(
            torch.nn.functional.relu(self.fusion(combined)))
        # logits = self.fc(fused)
        hidden = torch.nn.functional.relu(self.fc1(fused)) # trial
        logits = self.fc2(hidden) # trial
        # pred = torch.nn.functional.softmax(logits, dim=1)
        pred = logits # nn.CrossEntropyLoss expects raw logits as model output # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss = self.loss_fn(pred, label)
        return (pred, loss)

class MultimodalFakeNewsDetectionModelWithDialogue(pl.LightningModule):

    def __init__(self, hparams=None):
        super(MultimodalFakeNewsDetectionModelWithDialogue, self).__init__()
        if hparams:
            self.hparams.update(hparams) # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)
        self.dialogue_feature_dim = self.hparams.get("dialogue_feature_dim", self.text_feature_dim)

        self.model = self._build_model()

    # Required for pl.LightningModule
    def forward(self, text, image, dialogue, label):
        # pl.Lightning convention: forward() defines prediction for inference
        return self.model(text, image, dialogue,  label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        text, image, dialogue, label = batch["text"], batch["image"], batch["dialogue"], batch["label"]

        pred, loss = self.model(text, image, dialogue, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item())
        losses.append(loss.item())
        return loss

    # Optional for pl.LightningModule
    def training_step_end(self, batch_parts):
        """
        Aggregates results when training using a strategy that splits data
        from each batch across GPUs (e.g. data parallel)

        Note that training_step returns a loss, thus batch_parts returns a list
        of 2 loss values (since there are 2 GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        text, image, dialogue, label = batch["text"], batch["image"], batch["dialogue"], batch["label"]
        pred, loss = self.model(text, image, dialogue, label)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.sum(pred_label == label).item() / (len(label) * 1.0)
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda()
        }
        print(loss.item(), output['test_acc'])
        return output

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_acc': avg_accuracy
        }

        # pl.LightningModule has some issues displaying the results automatically
        # As a workaround, we can store the result logs as an attribute of the
        # class instance and display them manually at the end of testing
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
        self.test_results = logs

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_accuracy,
            'log': logs,
            'progress_bar': logs
        }

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9)
        return optimizer

    def _build_model(self):
        text_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.text_feature_dim)

        image_module = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        image_module.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=self.image_feature_dim)

        dialogue_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.dialogue_feature_dim)

        return JointTextImageDialogueModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            dialogue_module=dialogue_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            dialogue_feature_dim=self.dialogue_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        print("Training done...")
        global losses
        for loss_val in losses:
            print(loss_val)

# TODO: where does this go?
def build_image_transform(image_dim=224):
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

def get_checkpoint_filename_from_dir(path):
    return os.listdir(path)[0]

def test_out_dialogue_data():

    train_data_path = os.path.join(DATA_PATH, "multimodal_train_10000.tsv")
    test_data_path = os.path.join(DATA_PATH, "multimodal_test_1000.tsv")

    df = pd.read_pickle("./data/text_image_dialogue_dataframe.pkl")

    text_embedder = SentenceTransformer('all-mpnet-base-v2')
    image_transform = build_image_transform()
    train_dataset = MultimodalDataset(
        from_preprocessed_dataframe="./data/text_image_dialogue_dataframe.pkl",
        modality="text-image-dialogue",
        text_embedder=text_embedder,
        image_transform=image_transform,
        images_dir=IMAGES_DIR,
        num_classes=NUM_CLASSES
    )
    print(train_dataset)
    # print(train_dataset[0])
    print("train:", len(train_dataset))

    test_dataset = MultimodalDataset(
        from_preprocessed_dataframe="./data/test__text_image_dialogue_dataframe.pkl",
        modality="text-image-dialogue",
        text_embedder=text_embedder,
        image_transform=image_transform,
        images_dir=IMAGES_DIR,
        num_classes=NUM_CLASSES
    )
    print(test_dataset)
    # print(train_dataset[0])
    print("test:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=NUM_CPUS)
    test_loader = DataLoader(train_dataset, batch_size=16, num_workers=NUM_CPUS)

    hparams = {
        # "text_embedder": text_embedder,
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": NUM_CLASSES
    }
    model = MultimodalFakeNewsDetectionModelWithDialogue(hparams)
    trainer = None
    if torch.cuda.is_available():
        # Use all available GPUs with data parallel strategy
        callbacks = [PrintCallback()]
        trainer = pl.Trainer(
            # gpus=list(range(torch.cuda.device_count())),
            gpus=[1], # Someone is using GPU 0 and maxing out memory...
            strategy='dp',
            callbacks=callbacks,
            # enable_progress_bar=False, # Doesn't fix Batches progress bar issue
        )
    else:
        trainer = pl.Trainer()
    logging.debug(trainer)
    print(trainer)

    ## TRAINING
    trainer.fit(model, train_loader)

    ## EVALUATION
    # assets_version = "version_124"
    # checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    # checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    # print(checkpoint_path)
    # # NOTE: MAKE SURE YOU'RE USING THE WithDialogue MODEL!!!
    # model = MultimodalFakeNewsDetectionModelWithDialogue.load_from_checkpoint(checkpoint_path)
    # trainer.test(model, dataloaders=test_loader)
    # results = model.test_results
    # print(test_data_path)
    # print(checkpoint_path)
    # print(results)



if __name__ == "__main__":

    test_out_dialogue_data()
    assert(1 == 2)


    train_data_path = os.path.join(DATA_PATH, "multimodal_train_10000.tsv")
    test_data_path = os.path.join(DATA_PATH, "multimodal_test_1000.tsv")

    print("Using train data:", train_data_path)
    print("Using test data:", test_data_path)
    text_embedder = SentenceTransformer('all-mpnet-base-v2')
    image_transform = build_image_transform()
    train_dataset = MultimodalDataset(
        data_path=train_data_path,
        modality=MODALITY,
        text_embedder=text_embedder,
        image_transform=image_transform,
        images_dir=IMAGES_DIR,
        num_classes=NUM_CLASSES
    )
    test_dataset = MultimodalDataset(
        data_path=test_data_path,
        modality=MODALITY,
        text_embedder=text_embedder,
        image_transform=image_transform,
        images_dir=IMAGES_DIR,
        num_classes=NUM_CLASSES
    )
    print("train:", len(train_dataset))
    print("test: ", len(test_dataset))
    # print(train_dataset[0])

    ### DIALOGUE DATA PREPROCESSING
    # df = train_dataset.data_frame
    # print(df.columns)
    # print(df['id'])
    # # train_dataset._preprocess_dialogue()
    # train_dataset._preprocess_dialogue(from_saved_df_path="./data/comment_dataframe.pkl")

    # df = test_dataset.data_frame
    # # test_dataset._preprocess_dialogue() # Currently does not generate summaries if you don't pass from_saved_df_path
    # test_dataset._preprocess_dialogue(from_saved_df_path="./data/test__comment_dataframe.pkl")
    # ###

    ### DIALOGUE DATASET
    # dialogue_train_dataset = MultimodalDataset(
    #     data_path=train_data_path,
    #     modality="text-image-dialogue",
    #     text_embedder=text_embedder,
    #     image_transform=image_transform,
    #     images_dir=IMAGES_DIR,
    #     num_classes=NUM_CLASSES
    # )
    # dialogue_test_dataset = MultimodalDataset(
    #     data_path=test_data_path,
    #     modality="text-image-dialogue",
    #     text_embedder=text_embedder,
    #     image_transform=image_transform,
    #     images_dir=IMAGES_DIR,
    #     num_classes=NUM_CLASSES
    # )

    # hparams = {
    #     "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
    #     "num_classes": NUM_CLASSES
    # }
    # model = MultimodalFakeNewsDetectionModelWithDialogue(hparams)
    # print(model)
    # # assert(1 == 2)
    ###

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_CPUS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_CPUS)
    print(train_loader)

    hparams = {
        # "text_embedder": text_embedder,
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": NUM_CLASSES
    }
    model = MultimodalFakeNewsDetectionModel(hparams)

    trainer = None
    if torch.cuda.is_available():
        # Use all available GPUs with data parallel strategy
        callbacks = [PrintCallback()]
        trainer = pl.Trainer(
            gpus=list(range(torch.cuda.device_count())),
            strategy='dp',
            callbacks=callbacks,
            # enable_progress_bar=False, # Doesn't fix Batches progress bar issue
        )
    else:
        trainer = pl.Trainer()
    logging.debug(trainer)

    # TRAINING
    # trainer.fit(model, train_loader)

    # EVALUATION
    # # path_exp6 = os.path.join(PL_ASSETS_PATH, "version_70", "checkpoints", "epoch=15-step=4847.ckpt")
    # assets_version = "version_111"
    # checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    # checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    # print(checkpoint_path)
    # model = MultimodalFakeNewsDetectionModel.load_from_checkpoint(checkpoint_path)
    # trainer.test(model, dataloaders=test_loader)
    # results = model.test_results
    # print(test_data_path)
    # print(checkpoint_path)
    # print(results)
