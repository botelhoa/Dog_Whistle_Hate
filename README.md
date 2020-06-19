# Under Construction...

## This repository contains the code used to produce the results in _Deciphering Dog Whistles: Evaluating Automated Detection Algorithms for Multimodal Hate (Botelho 2020)_

Warning: Data contains sensitive content


## Table of Contents
* [Classification Models](#Classification-Models)
* [Model results](#Model-results)
* [Dog Whistle 5k dataset](#Dog-Whistle-5k-dataset)

## Classification Models
* Code for unimodal image model (InceptionResNetV2, NASNetLarge, Xception)
* Code for unimodal text model- LSTM
* Code for unimodal text model- Transformer (ALBERT, BERT, Electra, RoBERTa)
* Code for multimodal model
* Python implementation of best performing model

## Model Results
* Predictions and true labels for all uni and multi modal models
* Metrics for evaluating performance
* Error analysis

## Dog Whistle 5k Dataset
* Corpus of 5,000 annotated tweets subsampled from the [MMHS150k](https://gombru.github.io/2019/10/09/MMHS/) dataset annotated for Hate, Modality, Target, Strength, and Strategy (i.e. dog whistle or not)
* Codebook used to train annotation decisions
* Code used to collate annotation sheets into final dataset
* Visualizations of dataset contents
