# Deciphering Implicit Hate: Evaluating AutomatedDetection Algorithms for Multimodal Hate (Botelho, Vidgen, and Hale 2021)

This respository contains the code used to produce the results in _Deciphering Implicit Hate: Evaluating AutomatedDetection Algorithms for Multimodal Hate (Botelho, Vidgen, and Hale 2021)_. A special thanks to the Oxford Internet Institute and The Alan Turing Institute for their generous support.

Warning: Data contains sensitive content


## Table of Contents
* [Classification Models](#Classification-Models)
* [Data Analysis](#Data-Analysis)
* [Dog Whistle 5k Dataset](#Dog-Whistle-5k-Dataset)

## Classification Models
* Code for unimodal image model (InceptionResNetV2, NASNetLarge, Xception)
* Code for unimodal text model—LSTM
* Code for unimodal text model—Transformers (ALBERT, BERT, Electra, RoBERTa)
* Code for multimodal model (Feature Concatenation, MMBT)
* PyTorch implementation of best performing model

## Data Analysis
* Code used to collate annotation sheets into final ground truth dataset and assess annotator agreement levels
* Visualizations of dataset contents
* Predictions and true labels for all uni and multimodal models
* Model output analysis

## Dog Whistle 5k Dataset
* Corpus of 5,000 annotated tweets subsampled from the [MMHS150k](https://gombru.github.io/2019/10/09/MMHS/) dataset. Annotated for Hate, Modality, Target, Strength, and Strategy (i.e. dog whistle or not). Note: images to be downloaded from the original dataset
* Codebook used to train annotation decisions
* Train/Validation/Test data splits
