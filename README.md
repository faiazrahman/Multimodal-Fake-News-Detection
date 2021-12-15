# Multi-Modal Fine-Grained Fake News Detection with Dialogue Summarization

Faiaz Rahman
CS 677: Advanced Natural Language Processing
December 2021

## Setup

```
conda create --name mmfnd python=3.7
conda activate mmfnd
pip install -r requirements.txt
```

## Data

The text and dialogue (i.e. comments) data are included in the `data/` folder. However, given the size of the image data, it has not been copied over; the image data can be downloaded using the `data/image_downloader.py` script.

```
cd data
pip install -r requirements.txt
python image_downloader.py
```

## Running Experiments

The configuration files in `configs/` contain all of the parameters for running the experiments detailed in the paper. There is one config file per experiment in the paper, for a total of 12 configs. The data preprocessing step is run first separately (before both training and evaluation, for the train and test data, respectively). The same config file should be used for training and evaluation; it has been set up to use the appropriate parameters for each case.

### Training

```
python data_preprocessing.py --train --config <config_file_name>.yaml
python run_training.py --config <config_file_name>.yaml
```

Saved model assets (i.e. checkpoints) are stored in the `lightning_logs` folder. You will need to specify the version number (i.e. the most recent "version_*" folder that gets created when you start training) in the config file prior to running evaluation. (The model to evaluate is loaded from the most recent checkpoint in that specified version folder.)

### Evaluation

```
python data_preprocessing.py --test --config <config_file_name>.yaml
python run_evaluation.py --config <config_file_name>.yaml
```

Results are both logged using Python's logging module and are displayed as output.
