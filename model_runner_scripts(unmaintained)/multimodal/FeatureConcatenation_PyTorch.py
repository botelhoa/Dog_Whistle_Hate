#!/usr/bin/env python
# coding: utf-8

"""
This script trains and evaluates an MLP implemeneted in PyTorch. The model accepts a pre-extracted, concatenated mutlimodal feature vector as its input and outputs a classification decision.
"""


import pandas as pd
import numpy as np
import torch
import functools
import operator

from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer, AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm, trange



class MultimodalClassifier(nn.Module):
    def __init__(self, MLP_type, hidden_size: int=50, dropout: float=0.2, num_labels: int=4, input_len: int = 5120):
        """Initializes the network structure
        MLP_type: Which paper's MLP structure to use
        
        image_model: CovNet from Keras library to use as image feature extractor

        text_model: Transformer Model from Hugging Face to use as the text feature extractor

        hidden_size (int): Number of nodes in the hidden layer. Defaulted to 50. 

        dropout (float): Rate at which nodes are deactivated. Defaulted to 0.2. 
        
        num_labels (int): Number of labels to predict. Defaulted to 4.

        input_len (int): Length of input vector. Defaulted to 5120 (Image feature length (4096) + text feature length (1024)).
        """
        super(MultimodalClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if MLP_type == "Sabat":
            self.classifier = nn.Sequential(
                              nn.Linear(input_len, hidden_size),
                              nn.ReLU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden_size, hidden_size),
                              nn.ReLU(),
                              #nn.Dropout(dropout),
                              nn.Linear(hidden_size, num_labels)
                              #nn.Softmax(dim=1) 
                          )

        if MLP_type == "Gomez":
            self.classifier = nn.Sequential(
                              nn.Linear(input_len, input_len),
                              nn.BatchNorm1d(input_len),
                              nn.ReLU(),
                              #nn.Dropout(dropout),
                              nn.Linear(input_len, 1024),
                              nn.BatchNorm1d(1024),
                              nn.ReLU(),
                              #nn.Dropout(dropout),
                              nn.Linear(1024, 512),
                              nn.BatchNorm1d(512),
                              nn.ReLU(),
                              #nn.Dropout(dropout),
                              nn.Linear(512, num_labels),
                              nn.Softmax(dim=1) 
                          )


    def forward(self, features):
        """Initiaties foward pass through network
        
        features: Matrix of size number of tweets x 5120 containing concatenated image and text features

        """

        out = self.classifier(features.to(torch.float))

        return out


    def trainer(self, input_model, train_data, dev_data, early_stop_vals: dict, epochs: int = 25, learning_rate: float = 1e-5, weight_decay: float = 0.1, warmup: float = 0.06):   
            """
            Trains multimodal model

            input_model: Instatiation of model

            train_data: Dataloader object containing train data- image, text, labels

            dev_data: Dataloader object containing dev data- image, text, labels

            early_stopping: Dictionary containing patience value (int) and delta value (float). The patience determines the number of epochs to wait to achieve the given delta

            epochs (int): Number of times to run through all batches. Default value is 25.

            learning_rate (float): Default value is 1e-5.

            weight decay (float): Default value is 0.1 

            warmup (float): Default value is 0.06; percentage of training steps in warmup.
            """

            model = input_model.to(self.device)
            self.optimizer = optim.AdamW(model.classifier.parameters(), lr = learning_rate, weight_decay = weight_decay)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = warmup * (len(train_data) * epochs), num_training_steps = (1-warmup) * (len(train_data) * epochs))
            criterion = nn.CrossEntropyLoss().to(self.device)

            train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []

            for epoch in trange(epochs, desc= "Epoch"):
                if early_stopping(val_loss_values, early_stop_vals) == False:
                    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
                    print('Training...')

                    train_total_loss, train_total_len, train_num_correct = 0, 0, 0

                    model.train()

                    for step, batch in enumerate(train_data): 
                        if step % 50 == 0:
                            print("Processing batch...{} of {}".format(step, len(train_data)))

                        #model.zero_grad()
                        self.optimizer.zero_grad()  
                        batch_features, batch_labels, _ = tuple(t.to(self.device) for t in batch)
                        train_total_len += batch_features.shape[0]

                        logits = model(batch_features)

                        loss = criterion(logits, batch_labels).to(self.device) 
                        train_total_loss += loss

                        loss.backward() 
                        self.optimizer.step() 
                        self.scheduler.step()

                        pred = logits.argmax(1, keepdim=True).float()
                        correct_tensor = pred.eq(batch_labels.float().view_as(pred))
                        correct = np.squeeze(correct_tensor.cpu().numpy())
                        train_num_correct += np.sum(correct)

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


                    train_acc = train_num_correct / train_total_len
                    train_acc_values.append(train_acc)
                    avg_train_loss = train_total_loss / len(train_data)         
                    train_loss_values.append(avg_train_loss)

                    print()
                    print("Running Validation...")
                    print()

                    val_total_loss, val_total_len, val_num_correct = 0, 0, 0 

                    model.eval()

                    for batch in dev_data:
                        batch_features, batch_labels, _ = tuple(t.to(self.device) for t in batch)
                        val_total_len += batch_features.shape[0]

                        with torch.no_grad():        

                            logits = model(batch_features)
      
                        loss = criterion(logits, batch_labels) 
                        val_total_loss += loss
                      
                        pred = logits.argmax(1, keepdim=True).float()
                        correct_tensor = pred.eq(batch_labels.float().view_as(pred))
                        correct = np.squeeze(correct_tensor.cpu().numpy())
                        val_num_correct += np.sum(correct)

                    val_acc = val_num_correct / val_total_len
                    val_acc_values.append(val_acc)
                    avg_val_loss = val_total_loss / len(dev_data)
                    val_loss_values.append(avg_val_loss.cpu().numpy())

                    print("Epoch | Train Accuracy | Validation Accuracy | Training Loss | Validation Loss")
                    print(f"{epoch+1:3d} |    {train_acc:.3f}    |       {val_acc:.3f}       |    {avg_train_loss:.3f}    |     {avg_val_loss:.3f}")
                    print()


                    if epoch == (epochs-1):
                        training_plot(train_loss_values, val_loss_values)
                        training_dict = {"Train Accuracy": train_acc_values, "Train Loss": train_loss_values, "Val Accuracy": val_acc_values, "Val Loss": val_loss_values}
                        print("Training complete!")
                        return training_dict
                    else:
                        continue
          
                else:
                    print("Stopping early...")
                    training_plot(train_loss_values, val_loss_values)
                    training_dict = {"Train Accuracy": train_acc_values, "Train Loss": train_loss_values, "Val Accuracy": val_acc_valuess, "Val Loss": val_loss_values}
                    print("Training complete!")
                    return training_dict


    def test(self, input_model, test_data):
        """
        Tests the model's performance based on a several metrics

        input_model: Instatiation of model

        test_data: Dataloader object containing test data- image, text, labels
        """
   
        print('Predicting labels for {} sentences...'.format(len(test_data)))
        
        model = input_model.to(self.device)
        model.eval()

        predictions, true_labels, ids = [], [], []

        for batch in test_data:
            batch_features, batch_labels, batch_ids = tuple(t.to(self.device) for t in batch)
      
            with torch.no_grad():
                logits = model(batch_features)

            predictions.append(logits.detach().cpu().numpy())
            true_labels.append(batch_labels.to('cpu').numpy())
            ids.append(batch_ids.cpu().numpy())

        predictions = functools.reduce(operator.iconcat, predictions, [])
        true_labels = functools.reduce(operator.iconcat, true_labels, [])
        ids = functools.reduce(operator.iconcat, ids, [])

        print('    DONE.')
    
        return metrics(true_labels, predictions, argmax_needed= True), ids

    
    
#Hyperparamters
DROPOUT = 0.2
HIDDEN_SIZE = 100
BATCH_SIZE = 8
NUM_LABELS = 4
NUM_EPOCHS = 100
EARLY_STOPPING = {"patience": 5, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.1 
WARMUP = 0.06 
OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Multimodal/Feature Concatenation"



class DogWhistleDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return (self.data.shape[0])
    
    def __getitem__(self, i):
        features = np.array(self.data.iloc[0, 1:-2]) #start at 1 because of the Unnamed:0 header that gets added
        labels = self.data.loc[i, "labels"]
        ids = self.data.loc[i, "ids"]
        sample = (features, labels, ids)

        return sample

    
    
# Load data
# train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/combined_features.csv")
# dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/combined_features.csv")
# test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/combined_features.csv")

train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/combined_features_sabat.csv")
dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/combined_features_sabat.csv")
test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/combined_features_sabat.csv")

# Create dataset object
train_dataset = DogWhistleDataset(train)
dev_dataset = DogWhistleDataset(dev)
test_dataset = DogWhistleDataset(test)

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

Classifier = MultimodalClassifier("Sabat", HIDDEN_SIZE, DROPOUT, NUM_LABELS, 2048)
train_dict = Classifier.trainer(Classifier, train_dataloader, dev_dataloader, EARLY_STOPPING, 5, LEARNING_RATES[0], WEIGHT_DECAY, WARMUP)
(metric_vals, labels, preds), ids = Classifier.test(Classifier, test_dataloader)
#model_saver(Classifier, "Multimodal", OUTPUT_DIR, train_dict, labels, preds, metrics, ids)

print(metric_vals)

