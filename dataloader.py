import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"

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
        from_preprocessed_dataframe=None, # Preprocessed dataframe to load from
        from_dialogue_dataframe=None, # Partially preprocessed df to load from
        data_path=None, # Path to data (i.e. not using preprocessed dataframe),
        dir_to_save_dataframe="data", # Save the preprocessed dataframe here
        dataset_type="train",
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
                raise Exception("MultimodalDataset given invalid from_preprocessed_dataframe arg; \
                                 Must be path (str) to dataframe or pd.DataFrame")

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

        self.dataset_type = dataset_type
        self.dir_to_save_dataframe = dir_to_save_dataframe
        self.saved_dataframe_filename_prefix = ""
        if Modality(modality) == Modality.TEXT:
            self.saved_dataframe_filename_prefix = "text"
        elif Modality(modality) == Modality.IMAGE:
            self.saved_dataframe_filename_prefix = "image"
        elif Modality(modality) == Modality.TEXT_IMAGE:
            self.saved_dataframe_filename_prefix = "text_image"
        elif Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
            self.saved_dataframe_filename_prefix = "text_image_dialogue"

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

        if Modality(self.modality) in [Modality.TEXT, Modality.TEXT_IMAGE, \
                                       Modality.TEXT_IMAGE_DIALOGUE]:
            text = self.data_frame.loc[idx, 'clean_title']
            text = self.text_embedder.encode(text, convert_to_tensor=True)
            item["text"] = text
        if Modality(self.modality) in [Modality.IMAGE, Modality.TEXT_IMAGE, \
                                       Modality.TEXT_IMAGE_DIALOGUE]:
            image_path = os.path.join(IMAGES_DIR, item_id + IMAGE_EXTENSION)
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)
            item["image"] = image
        if Modality(self.modality) == Modality.TEXT_IMAGE_DIALOGUE:
            dialogue = self.data_frame.loc[idx, 'comment_summary']
            dialogue = self.text_embedder.encode(dialogue, convert_to_tensor=True)
            item["dialogue"] = dialogue

        return item

    def _preprocess_df(self, df):
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

        # Save this dataframe into a pickle
        # Filename will look something like "train__text_image_dialogue__dataframe.pkl"
        filename = "__".join([self.dataset_type, self.saved_dataframe_filename_prefix, "dataframe.pkl"])
        save_path = os.path.join(self.dir_to_save_dataframe, filename)
        df.to_pickle(save_path)
        print("Preprocessed dataframe saved to {}".format(save_path))
        logging.info("Preprocessed dataframe saved to {}".format(save_path))

        return df

    def _preprocess_dialogue(self, from_saved_df_path=""):
        """ A comment's 'submission_id' is linked (i.e. equal) to a post's 'id'
        and 'body' contains the comment text and 'ups' contains upvotes """

        # Save this dialogue dataframe into a pickle
        # Filename will look something like "train__text_image_dialogue__dialogue_dataframe.pkl"
        filename = "__".join([self.dataset_type, self.saved_dataframe_filename_prefix, "dialogue_dataframe.pkl"])
        save_path = os.path.join(self.dir_to_save_dataframe, filename)

        def generate_summaries_and_save_df(df, save_path="data"):
            # Add new column in main dataframe to hold dialogue summaries
            self.data_frame['comment_summary'] = ""

            failed_ids = []
            for iteration, text_id in enumerate(self.text_ids):
                if (iteration % 250 == 0):
                    print("Generating summaries for item {}...".format(iteration))
                    # Save progress so far
                    self.data_frame.to_pickle(save_path)

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

            # Save final dialogue dataframe
            self.data_frame.to_pickle(save_path)
            print("Preprocessed dialogue dataframe saved to {}".format(save_path))
            logging.info("Preprocessed dialogue dataframe saved to {}".format(save_path))

            logging.debug(self.data_frame)
            logging.debug(self.data_frame['comment_summary'])
            logging.debug("num_failed:", len(failed_ids))
            logging.debug(failed_ids)

        if from_saved_df_path != "":
            # Special Case (see above comment in __init__)
            df = pd.read_pickle(from_saved_df_path)
            generate_summaries_and_save_df(df, save_path=save_path)
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

            # Save dataframe so far before summary generation
            df.to_pickle(save_path)

            generate_summaries_and_save_df(df, save_path=save_path)
