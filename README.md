# Multi-Modal Fine-Grained Fake News Detection with Dialogue Summarization

_This project was developed by Faiaz Rahman originally for CS 677: Advanced Natural Language Processing under Dr. Dragomir Radev at Yale University._

## Setup

We recommend using a virtual environment via Conda. We have provided two requirements files. `distilled_requirements.txt` contains the essential dependencies based on imports, as determined by `pipreqs`. `requirements.txt` contains an exact `pip freeze` of all the dependencies in the virtual environment used when running the experiments in the paper on Yale's Tangra computing cluster. We recommend trying `distilled_requirements.txt` first, but if dependency issues persist, to use the bulkier `requirements.txt`.

```
conda create --name mmfnd python=3.7
conda activate mmfnd
pip install -r [requirements.txt | distilled_requirements.txt]
```

## Data

Fakeddit (Nakamura et al., 2020) is a multi-modal dataset consisting of over 1 million samples from multiple categories of fake news, labeled with 2-way, 3-way, and 6-way classification categories to allow for both binary classification and, more interestingly, fine-grained classification. The dataset can be downloaded from the original [Fakeddit repo](https://github.com/entitize/Fakeddit). The text and comments data can be downloaded directly and `cp`'d into the `data/` folder. Our codebase has a slightly modified version of their image downloading script `data/image_downloader.py`, which can be used when downloading image data.

```
cd data
pip install -r requirements.txt
python image_downloader.py
```

## Running Experiments

We run experiments to compare (1) the performance of text-image multi-modal models with text- image-dialogue multi-modal models, and (2) the performance of different text encoder models. We do not compare with single-modal models, since Nakamura et al. (2020) already compared text-image multi-modal models with single-modal text and single-modal image models and found that the multi-modal approach indeed had better performance. Thus, we focus on quantifying the performance of including dialogue data via dialogue summarization.

The configuration files in `configs/` contain all of the parameters for running the experiments detailed in the paper. There is one config file per experiment in the paper, for a total of 12 configs. The data preprocessing step is run first separately (before both training and evaluation, for the train and test data, respectively). The same config file should be used for training and evaluation; it has been set up to use the appropriate parameters for each case.

### Training

For training, to allow for flexible hyperparameter tuning, you can use a specified config file (which contains the settings for a specific experiment) but then also override specific hyperparameters via command-line arguments. For example, if you run `python run_traiing --config some_config.yaml --learning_rate 0.001`, it will use a learning rate of `0.001` (regardless of whatever learning rate is specified in the config), and then for all other hyperparameters, it will use the value in the config file.

Note that if neither a config file nor command-line args are specified, the default values in the `run_training.py` script will be used.

```
python data_preprocessing.py --train --config <config_file_name>.yaml
python run_training.py --config <config_file_name>.yaml
        [--modality text | image | text-image | text-image-dialogue]
        [--num_classes 2 | 3 | 6]
        [--batch_size (int)]
        [--learning_rate (float)]
        [--num_epochs (int)]
        [--dropout_p (float)]
        [--text_embedder all-mpnet-base-v2 | all-distilroberta-v1]
        [--dialogue_summarization_model sshleifer/distilbart-cnn-12-6 | bart-large-cnn | t5-small | t5-base | t5-large]
        [--gpus 0 | 1 | 0,1 | 0,1,2,3 | etc.]
        [--train_data_path (str)]
        [--test_data_path (str)]
```

Saved model assets (i.e. checkpoints) are stored in the `lightning_logs` folder. You will need to specify the version number (i.e. the most recent "version_*" folder that gets created when you start training) in the config file prior to running evaluation. (The model to evaluate is loaded from the most recent checkpoint in that specified version folder.)

### Evaluation

Once you have completed hyperparameter tuning, create a config file with your experiment settings, which you will then reuse during evaluation. Run training with this config file to produce the trained model assets. If using the default `lightning_logs/` folder to save model assets, you can note the `version_*` in the most recent folder that is created after training starts. You can specify that version number as `trained_model_version` in your config file. If you trained model has some other filename or path, you can specify that as `trained_model_path` in your config file.

```
python data_preprocessing.py --test --config <config_file_name>.yaml
python run_evaluation.py --config <config_file_name>.yaml
```

Results are both logged using Python's logging module and are displayed as output.

### Models

For our text encoder, we experiment with both RoBERTa and MPNet. Specifically, our implementation uses `all-distilroberta-v1` and `all-mpnet-base-v2` from `sentence-transformers`.
For our image module, we use ResNet-152 via `torchvision.models.resnet152`.
For our dialogue summarization, we use a BART summarization pipeline from HuggingFace’s `transformers`, specifically with the `sshleifer/distilbart-cnn-12-6` model. For encoding the generated dialogue summary, we use the same models as the text encoder (i.e., RoBERTa and MPNet).

### Experiment Settings

We train on two NVIDIA K80 GPUs with Driver version 465.19.01 and CUDA version 11.3. We use PyTorch (version 1.10.0) to implement our models, including using PyTorch Lightning for our model training and evaluation. We run training and evaluation on both GPUs in data parallel (i.e., splitting each batch across both GPUs), with a batch size of 32 (and thus each GPU processing 16 items per batch, with the root node aggregating the results). We use a learning rate of `1e-4`, dropout percentage of 0.1, and train for 5 epochs. We use Adam as our optimizer and cross-entropy as our loss function.

We ran hyperparameter tuning and found that learning rates of `1e-3` and `1e-5` caused the loss to not decrease. We also found that using SGD as our optimizer (both with and without momentum) caused the loss to not decrease.

Our results are presented in the paper, which is also available in this repo at `./paper.pdf`.
