# -*- coding: utf-8 -*-
"""Transformer_Runner.ipynb

This py script fine-tunes, tests, and optimizes over a hyperparameter sweep various classification models 
provided through the Hugging Face library including text pre-processing.
It is compatible with AlBERT, BART, BERT, DistilBERT, RoBERTa, and XLNet.
Inspiration was derived from https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

"""

import pandas as pd
import os
import numpy as np
import torch
import functools
import operator

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer, AlbertForSequenceClassification, AlbertTokenizer, ElectraForSequenceClassification, ElectraTokenizer, get_linear_schedule_with_warmup
from transformers import BartForSequenceClassification, BartTokenizer, XLNetForSequenceClassification, XLNetTokenizer
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict, Counter
from tqdm import tqdm, trange
from helperfunctions import *



#Load data for pretraining
# df = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/labeled_data.csv", encoding='utf-8')
# train, _, dev = stratify(df, "hate_speech", [0.9, 0.1])

train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/dog_whistle_train.csv", encoding='utf-8')
dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/dog_whistle_dev.csv", encoding='utf-8')
test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/dog_whistle_test.csv", encoding='utf-8')

train["combined_text"] = combine_text(train)
dev["combined_text"] = combine_text(dev)
test["combined_text"] = combine_text(test)

train = train[["image_number", "combined_text", "Unimodal_text_numeric", "Primary_numeric_gt"]]
dev = dev[["image_number", "combined_text", "Unimodal_text_numeric", "Primary_numeric_gt"]]
test = test[["image_number", "combined_text", "Unimodal_text_numeric", "Primary_numeric_gt"]]


#visualize word count distribution to inform pad_length
# from matplotlib import pyplot as plt
# import seaborn as sns

# tweet_lens = []

# for i in train["combined_text"]:
#     tokens = i.split()
#     tweet_lens.append(len(tokens))

# fig = plt.figure(figsize=(10, 8))
# g1 = sns.distplot(tweet_lens)

# g1.set_xlabel('Tweet Word Count')
# g1.set_ylabel('Frequency Density')

# fig.savefig('tweet_length_distribution.png',bbox_inches='tight')

# train["lengths"] = tweet_lens

# a = train[train["lengths"] > 250]
# len(a)



class TransformerClassifier(nn.Module):
    def __init__(self, model, num_labels: int, pad_length: int= 200, batch_size: int = 32):
        """
        model: From HuggingFace transformers library

        num_labels (int): Number of annotation classes

        pad_length (int): Max sentence length. Defaulted to 64.

        batch_size (int): Number of sentences in batch. Default is 32.
        """
        super(TransformerClassifier, self).__init__()
        self.model_types = {
          "AlBERT": [AlbertForSequenceClassification, AlbertTokenizer, 'albert-xlarge-v2'],  # 'albert-xlarge-v2' "albert-large-v2" 'albert-base-v2' 'albert-xxlarge-v2'
          "BART": [BartForSequenceClassification, BartTokenizer, "bart-large"], 
          "BERT": [BertForSequenceClassification, BertTokenizer, 'bert-large-uncased'], #'bert-base-uncased'
          "DistilBERT": [DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-cased'], # 'distilbert-base-cased']
          "Electra": [ElectraForSequenceClassification, ElectraTokenizer, "google/electra-large-discriminator"], #"google/electra-small-discriminator"
          "RoBERTa": [RobertaForSequenceClassification, RobertaTokenizer, 'roberta-large'], #'roberta-base'
          "XLNet": [XLNetForSequenceClassification, XLNetTokenizer, "xlnet-large-cased"], #"xlnet-base-cased"
          } 
        self.model_selection = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model_types[self.model_selection][0].from_pretrained(self.model_types[self.model_selection][2], num_labels = num_labels).to(self.device)
        self.seed_val = 22
        self.pad_length = pad_length
        self.batch_size = batch_size

        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed_val)

    def preprocessor(self, sequence, labels):
        """
        This function converts a string of text into a tokenized format compatible with the selected model

        sequence: An iterable series of data (i.e. Pandas Series, list..) where elements are strings

        labels: Pandas series containing data annotations
        """
        
        self.tokenizer = self.model_types[self.model_selection][1].from_pretrained(self.model_types[self.model_selection][2])

        indexed_tokens = []
        attention_masks = []


        for counter, sentence in enumerate(sequence):
            if counter % 1000 == 0:
                print("Processing row {}".format(counter))
            if counter == len(sequence):
                print("Done!")


            encoded_dict = self.tokenizer.encode_plus(
                      sentence,            
                      add_special_tokens = True,
                      max_length = self.pad_length,         
                      pad_to_max_length = True,
                      return_attention_mask = True,  
                      return_tensors = 'pt',   
                  )
  
            indexed_tokens.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        return self.batcher(torch.cat(indexed_tokens, dim=0), torch.cat(attention_masks, dim=0), labels)


    def batcher(self, indexed_tokens, attention_masks, labels):
        """
        This function creates batches of a specified size to save on memory

        indexed_tokens: Tokenized text output by model preprocesser

        attention_masks: Attention masks output by model preprocesser

        labels: Pandas series containing data annotations
        """

        data = TensorDataset(indexed_tokens, attention_masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size = self.batch_size)

        return dataloader


    def fine_tune(self, train_data, train_labels, dev_data, dev_labels, normalize_list, annotate_list, early_stop_vals: dict, epochs: int = 3, learning_rate: float = 2e-5, weight_decay: float = 0.1, warmup: float = 0.06):   
        """
        Updates pre-trained model's weights based on given dataset

        train_data: Pandas series object containing text data for train set

        train_labels: Pandas series object containing ground truth annotations for train set

        dev_data: Pandas series object containing text data for dev set

        dev_labels: Pandas series object containing ground truth annotations for dev set

        normalize_list: list of data features to clean

        annotate_list: list of data features to annotate

        early_stopping: Dictionary containing patience value (int) and delta value (float). The patience determines the number of epochs to wait to achieve the given delta

        epochs (int): Number of times to run through all batches. Default value is 3 according to 2-4 recommended in original BERT paper.

        learning_rate (float): Default value is 2e-5 according to recommended value from original BERT paper.

        weight decay (float): Default value is 0.1 

        warmup (float): Default value is 0.06; percentage of training steps in warmup
        """
    
        self.train_labels = torch.Tensor(train_labels.values).to(torch.int64)
        self.dev_labels = torch.Tensor(dev_labels.values).to(torch.int64)
        
        clean_train_data = clean_text(train_data, normalize_list, annotate_list)
        clean_val_data = clean_text(dev_data, normalize_list, annotate_list)
        self.train_dataloader = self.preprocessor(clean_train_data, self.train_labels)
        self.val_dataloader = self.preprocessor(clean_val_data, self.dev_labels)

        self.optimizer = optim.AdamW(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = warmup * (len(self.train_dataloader) * epochs), num_training_steps = (1-warmup) * (len(self.train_dataloader) * epochs))

        train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []

        for epoch in trange(epochs, desc= "Epoch"):
            if early_stopping(val_loss_values, early_stop_vals) == False:
                print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
                print('Training...')

                train_total_loss, train_total_len, train_num_correct = 0, 0, 0

                self.model.train()

                for step, batch in enumerate(self.train_dataloader):
                    if step % 50 == 0:
                        print("Processing batch...{}".format(step))
                        print("  Batch {:>5,}  of  {:>5,}.".format(step, len(self.train_dataloader)))

                    train_total_len += batch[0].shape[0]
                    b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

                    self.model.zero_grad()  

                    outputs = self.model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels.unsqueeze(0))

                    train_total_loss += outputs[0].item()
                    outputs[0].backward()

                    pred = outputs[1].argmax(1, keepdim=True).float()
                    correct_tensor = pred.eq(b_labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.cpu().numpy())
                    train_num_correct += np.sum(correct)

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                train_acc = train_num_correct / train_total_len
                train_acc_values.append(train_acc)
                avg_train_loss = train_total_loss / len(self.train_dataloader)            
                train_loss_values.append(avg_train_loss)

                print()
                print("  Average training loss: {0:.2f}".format(avg_train_loss))

                print()
                print("Running Validation...")
                print()

                val_total_loss, val_total_len, num_correct = 0, 0, 0 

                self.model.eval()

                for batch in self.val_dataloader:
                    val_total_len += batch[0].shape[0]
                    b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

                    with torch.no_grad():        

                        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
  
                    val_total_loss += outputs[0].item()

                    pred = outputs[1].argmax(1, keepdim=True).float()
                    correct_tensor = pred.eq(b_labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.cpu().numpy())
                    num_correct += np.sum(correct)

                val_acc = num_correct / val_total_len
                val_acc_values.append(val_acc)
                avg_val_loss = val_total_loss / len(self.val_dataloader)  
                val_loss_values.append(avg_val_loss)

                print(f"Epoch | Validation Accuracy | Training Loss | Validation Loss")
                print(f"{epoch+1:5d} |       {val_acc:.5f}       |    {avg_train_loss:.5f}    |     {avg_val_loss:.5f}")

                print()

                if epoch == (epochs-1):
                    training_plot(train_loss_values, val_loss_values)
                    training_dict = {"Train Accuracy": train_acc_values, "Train Loss": train_loss_values, "Val Accuracy": val_acc_values, "Val Loss": val_loss_values}
                    print("Training complete!")
                    return training_dict, self.tokenizer
                else:
                    continue
      
            else:
                print("Stopping early...")
                training_plot(train_loss_values, val_loss_values)
                training_dict = {"Train Accuracy": train_acc_values, "Train Loss": train_loss_values, "Val Accuracy": val_acc_values, "Val Loss": val_loss_values}
                print("Training complete!")
                return training_dict, self.tokenizer


    def test(self, test_data, test_labels, normalize_list, annotate_list):
        """
        Tests the model's performance based on a several metrics

        test_data: Pandas series object containing text data

        test_labels: Pandas series object containing labels

        normalize_list: list of data features to clean

        annotate_list: list of data features to annotate
        """
        self.test_labels = torch.Tensor(test_labels.values).to(torch.int64)

        clean_test_data = clean_text(test_data, normalize_list, annotate_list)
        self.test_dataloader = self.preprocessor(clean_test_data, self.test_labels)
    
        print('Predicting labels for {} sentences...'.format(len(self.test_labels)))

        self.model.eval()

        predictions, true_labels = [], []

        for batch in self.test_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)
      
            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)

            logits = outputs[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')

        predictions = functools.reduce(operator.iconcat, predictions, [])
        true_labels = functools.reduce(operator.iconcat, true_labels, [])
    
        return metrics(true_labels, predictions, argmax_needed= True)


    def save(self, output_directory, training_dict, labels, preds, results):
        """
        Saves PyTorch Model and other outputs
        
        output_directory: Directory to folder to save file in

        training_dict: Dictionary of training and validation values 

        labels: List of labels for test set

        preds: List of model predictions after passed through argmax()

        results: Dictionary of metrics

        tokenizer: Tokenizer to be saved. Defaulted to None.
        """

        output_directory = os.path.join(output_directory, self.model_selection)
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        os.chdir(output_directory)

        np.save(self.model_selection+"_dogwhistle_train_results.npy", training_dict) #save training dict
        np.save(self.model_selection+"_dogwhistle_test_results.npy", results) #save test metrics
        
        test_predictions = pd.DataFrame([labels, preds]) #save predictions and labels 
        test_predictions = test_predictions.T
        test_predictions = test_predictions.rename(columns={0: 'Labels', 1: 'Predictions'})
        test_predictions.to_csv(self.model_selection+"_dogwhistle_predictions.csv")

        #save models
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model 
        model_to_save.save_pretrained(output_directory) 
        tokenizer.save_pretrained(output_directory)

        return print("Saving complete.")

    

#RoBERTa fine-tuning hyperparameters for GLUE: 
NORMALIZE_LIST = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']
ANNOTATE_LIST = ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']
LEARNING_RATE = [1e-5, 2e-5, 3e-5]
N_EPOCHS = 10 
EARLY_STOPPING = {"patience": 2, "delta": 0.005} 
N_LABELS =  4
PAD_LENGTH = 250
BATCH_SIZE = [16, 32, 64]
WEIGHT_DECAY = 0.1 
WARMUP = 0.06 
OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text"

# #Pretrain models
#Classifier = TransformerClassifier("DistilBERT", N_LABELS, PAD_LENGTH, 16) 
#train_dict, tokenizer = Classifier.fine_tune(train["tweet"], train["hate_speech"], dev["tweet"], dev["hate_speech"], NORMALIZE_LIST, ANNOTATE_LIST, EARLY_STOPPING, N_EPOCHS, 1e-5, WEIGHT_DECAY, WARMUP) 
#results, labels, preds = Classifier.test(test["combined_text"], test["Primary_numeric_gt"], NORMALIZE_LIST, ANNOTATE_LIST)
#Classifier.save(OUTPUT_DIR, train_dict, [], [], {})



#Practice run
# Classifier = TransformerClassifier("BERT", N_LABELS, PAD_LENGTH, 16) 
# train_dict, tokenizer = Classifier.fine_tune(train["combined_text"], train["Unimodal_text_numeric"], dev["combined_text"], dev["Unimodal_text_numeric"], NORMALIZE_LIST, ANNOTATE_LIST, EARLY_STOPPING, N_EPOCHS, 1e-5, WEIGHT_DECAY, WARMUP) 
# results, labels, preds = Classifier.test(test["combined_text"], test["Primary_numeric_gt"], NORMALIZE_LIST, ANNOTATE_LIST)
# Classifier.save(OUTPUT_DIR, train_dict, labels, preds, results)
# results


#Training loop
results_dict = {}
max_f1_value = 0

for i in BATCH_SIZE:
    learning_rate_dict = {}
    for j in LEARNING_RATE: 
        Classifier = TransformerClassifier("DistilBERT", N_LABELS, PAD_LENGTH, i) 
        train_dict, tokenizer = Classifier.fine_tune(train["combined_text"], train["Unimodal_text_numeric"], dev["combined_text"], dev["Unimodal_text_numeric"], NORMALIZE_LIST, ANNOTATE_LIST, EARLY_STOPPING, N_EPOCHS, j, WEIGHT_DECAY, WARMUP) 
        learning_rate_dict[j], labels, preds = Classifier.test(test["combined_text"], test["Primary_numeric_gt"], NORMALIZE_LIST, ANNOTATE_LIST)
        results2, labels2, preds2 = Classifier.test(test["combined_text"], test["Unimodal_text_numeric"], NORMALIZE_LIST, ANNOTATE_LIST)

    if learning_rate_dict[j]["f1"] >= max_f1_value: #only save best model
        max_f1_value = learning_rate_dict[j]["f1"]
        print("The new top F1 score is: {}. Saving model...".format(max_f1_value))
        #model_saver(Classifier, "BERT", OUTPUT_DIR, train_dict, labels, preds, learning_rate_dict[j], tokenizer)
        Classifier.save(OUTPUT_DIR, train_dict, labels, preds, learning_rate_dict[j])
        np.save(os.path.join(os.path.join(OUTPUT_DIR, "DistilBERT"), "dogwhistle_unimodal_label_test_results.npy"), results2)

    results_dict[i] = learning_rate_dict 

#save complete training results
np.save(os.path.join(os.path.join(OUTPUT_DIR, "DistilBERT"), "dogwhistle_total_training_results.npy"), results_dict)


#Apply hyperparameters
BEST_LR = 1e-5
BEST_BATCH_SIZE = 16
MODEL_LIST = ["AlBERT", "BERT", "Electra", "RoBERTa"]

model_comparison = {}

for model in MODEL_LIST:
    Classifier = TransformerClassifier(model, N_LABELS, PAD_LENGTH, BEST_BATCH_SIZE) 
    train_dict, tokenizer = Classifier.fine_tune(train["combined_text"], train["Unimodal_text_numeric"], dev["combined_text"], dev["Unimodal_text_numeric"], NORMALIZE_LIST, ANNOTATE_LIST, EARLY_STOPPING, N_EPOCHS, BEST_LR, WEIGHT_DECAY, WARMUP) 
    results1, labels1, preds1 = Classifier.test(test["combined_text"], test["Primary_numeric_gt"], NORMALIZE_LIST, ANNOTATE_LIST)
    results2, labels2, preds2 = Classifier.test(test["combined_text"], test["Unimodal_text_numeric"], NORMALIZE_LIST, ANNOTATE_LIST)
    model_comparison[model] = {"Unimodal labels": results2, "Multumodal labels": results1}

    #model_saver(Classifier, model, OUTPUT_DIR, train_dict, labels, preds, model_comparison[model], tokenizer) #Update

#save complete training results
np.save(os.path.join(os.path.join(OUTPUT_DIR, model), "dogwhistle_total_training_results.npy"), model_comparison)



#Visualize confusion matrix
X_TICK_LABELS = ["None", "Hateful", "Counter-speech", "Reclaimed"]
Y_TICK_LABELS = ["None", "Hateful", "Counter-speech", "Reclaimed"]
COLOR = "Blues"
SAVE_NAME = "_cm_dogwhistle.png" #update with best model
BEST_RESULTS_CM = 

confusion_matrix_plotter(BEST_RESULTS_CM, SAVE_NAME, X_TICK_LABELS, Y_TICK_LABELS, COLOR)

