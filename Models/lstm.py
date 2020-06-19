# -*- coding: utf-8 -*-
"""LSTM.ipynb

This py script trains, tests, and optimizes over a hyperpararmeter sweep a bi-LSTM
with fastText embeddings including performing text pre-processing 

"""

import pandas as pd
import os
import numpy as np
import torch

from torch import nn, optim
from torchtext.data import Field, Dataset, Example, BucketIterator, Iterator, TabularDataset
from tqdm import tqdm, trange
from helperfunctions import *




"""### Prepare Data"""

NORMALIZE_LIST = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']
ANNOTATE_LIST = ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']


# Load data
train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/dog_whistle_train.csv", encoding='utf-8')
dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/dog_whistle_dev.csv", encoding='utf-8')
test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/dog_whistle_test.csv", encoding='utf-8')

# Combine Image and Tweet Text
train["text"] = combine_text(train)
dev["text"] = combine_text(dev)
test["text"] = combine_text(test)

# Clean data
train["text"] = clean_text(train["text"], NORMALIZE_LIST, ANNOTATE_LIST)
dev["text"] = clean_text(dev["text"], NORMALIZE_LIST, ANNOTATE_LIST)
test["text"] = clean_text(test["text"], NORMALIZE_LIST, ANNOTATE_LIST)

# Rename label column 
train["labels"] = train["Unimodal_text_numeric"]
dev["labels"] = dev["Unimodal_text_numeric"]
test["labels"] = test["Primary_numeric_gt"]
#test["labels"] = test["Unimodal_text_numeric"]

# Subset with necessary data
train_use = train[["text", "labels", "image_number"]]
dev_use = dev[["text", "labels", "image_number"]]
test_use = test[["text", "labels", "image_number"]]

# Define fields
TEXT = Field(sequential=True, tokenize="spacy", include_lengths=True) 
LABEL = Field(sequential=False, use_vocab=False)
IDS = Field(sequential=False, use_vocab=False)
fields = [('text', TEXT), ('labels', LABEL), ('image_number', IDS)]

# Create datasets
train_set = Dataset([Example.fromlist(i, fields) for i in train_use.values.tolist()], fields=fields)
dev_set = Dataset([Example.fromlist(i, fields) for i in dev_use.values.tolist()], fields=fields)
test_set = Dataset([Example.fromlist(i, fields) for i in test_use.values.tolist()], fields=fields)

# Get training words and pretrained vectors (training words without vectors are initialized randomly)
TEXT.build_vocab(train_set, vectors='fasttext.simple.300d', unk_init=torch.Tensor.normal_)




"""## Model Setup"""

class LSTMClassifier(nn.Module):
    def __init__(self, train, dev, test, pad_id, input_dim: int, embedding_dim: int=300, hidden_dim: int=256, output_dim: int=4, n_layers: int=2, dropout: float=0.2, bidirectional: bool=True, batch_size: int=64):
        """
        train: dataset object of training data

        dev: dataset object of dev data

        test: dataset object of test data
        
        pad_id: pads output with embedding vector
        
        input_dim (int): length of text
        
        embedding_dim (int): number of embedding dimensions defaulted to 300
        
        hidden_dim (int): number of hidden nodes defaulted to 256
        
        output_dim (int): number of labels in classification task defaulted to 2
        
        n_layers (int): number of recurrent layers defaulted to 2
        
        dropout (float): percent on nodes turned off during training defaulted to 0.2

        bidirectional (boolean): whether bidirectional layers will be added, defaulted to True

        batch_size (int): size of mini batches
        """
        
        super(LSTMClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_iter, self.dev_iter, self.test_iter = BucketIterator.splits((train, dev, test), 
                                                                                batch_size=batch_size, 
                                                                                device=self.device, 
                                                                                sort_key=lambda x: len(x.text), 
                                                                                sort_within_batch=True
                                                                                )

        self.output_dim = output_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sig = nn.Sigmoid()


    def forward(self, text, lengths):
        """
        This function sets up the model's foward pass

        Initiates forward pass of data
        
        text: vector representation of text string
        
        lengths: second element in bucket iterator
        """

        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths) #,batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        outputs=self.fc(hidden)
        
        return outputs
      
    
    def trainer(self, model, learning_rate: float, early_stop_vals, epochs: int=50):
        """
        This function trains the model

        model: Instantiation of model

        learning_rate (float): determines steps size while minimizing loss function
        
        early_stop_vals: Dictionary containing patience level and minimum improvement delta
        
        epochs (int): Number of training epochs defaulted to 50
        """

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        model = model.to(self.device)
        
        train_scores, train_losses, val_scores, val_losses, = [], [], [], []
        
        for epoch in trange(epochs, desc= "Epoch"):
            print("Processing epoch {}".format(epoch+1))
            if early_stopping(val_losses, early_stop_vals) == False:

                model.train()

                train_acc = 0
                batch_train_losses = []

                for step, batch in enumerate(self.train_iter): # Loop over mini-batches

                    optimizer.zero_grad()

                    text, lengths = batch.text

                    if self.output_dim ==  1:
                        predictions = model(text, lengths).squeeze()
                        criterion = nn.BCELogits().to(self.device)
                        loss = criterion(predictions, batch.labels.to(self.device, dtype=torch.float)) 
                        loss.backward() # Backpropagate loss
                        optimizer.step() # Update weights
                        train_acc += self.binary_accuracy(predictions, batch.labels).item()
                        batch_train_losses.append(loss.tolist())
                    else:
                        predictions = model(text, lengths)
                        criterion = nn.CrossEntropyLoss().to(self.device)
                        loss = criterion(predictions, batch.labels) 
                        loss.backward() # Backpropagate loss
                        optimizer.step() # Update weights
                        train_acc += self.batch_accuracy(predictions, batch.labels).item() 
                        batch_train_losses.append(loss.tolist())


                batch_loss = sum(batch_train_losses)/len(batch_train_losses)
                train_losses.append(batch_loss)
                train_acc /= len(self.train_iter)
                train_scores.append(train_acc)
                print('Accuracy on train data:\t{:.4f}'.format(train_acc))
                model.eval() # Compute accuracy on validation data

                val_acc = 0
                batch_val_losses = []

                with torch.no_grad():

                    for batch in self.dev_iter:

                        text, lengths = batch.text

                        if self.output_dim ==  1:
                            preds = model(text, lengths).squeeze()
                            criterion = nn.BCELogits().to(self.device)
                            loss = criterion(preds, batch.labels.to(self.device, dtype=torch.float)) 
                            batch_val_losses.append(loss.tolist())
                            val_acc += self.binary_accuracy(preds, batch.label).item()
                        else:
                            preds = model(text, lengths)
                            criterion = nn.CrossEntropyLoss().to(self.device)
                            loss = criterion(preds, batch.labels) 
                            batch_val_losses.append(loss.tolist())
                            val_acc += self.batch_accuracy(preds, batch.labels).item()
                        

                batch_loss = sum(batch_val_losses)/len(batch_val_losses)
                print("Batch dev loss: {}".format(batch_loss))
                val_losses.append(batch_loss)
                val_acc /= len(self.dev_iter)
                val_scores.append(val_acc)

                print('Accuracy on dev data:\t{:.4f}'.format(val_acc))
                
                if epoch == (epochs-1):
                    training_plot(train_losses, val_losses)
                    training_dict = {"Train Accuracy": train_scores, "Train Loss": train_losses, "Val Accuracy": val_scores, "Val Loss": val_losses}
                    print("Training complete!")
                    return training_dict
                else:
                    continue
                
            else:
                print("Stopping early...")
                print("Training complete!")
                training_plot(train_losses, val_losses)
                training_dict = {"Train Accuracy": train_scores, "Train Loss": train_losses, "Val Accuracy": val_scores, "Val Loss": val_losses}
                print("Training complete!")
                return training_dict


    def test(self, model):
        """
        This function performs a forward pass on the test data and computes the performance metrics

        model: instantiation of model
        """
        
        model = model.to(self.device)
        model.eval()

        labels = list()
        preds = list()
        ids = list()

        with torch.no_grad():

            for batch in self.test_iter:

                text, lengths = batch.text
                output = model(text, lengths)

                preds.extend([p.item() for p in output.argmax(dim=1, keepdim=True)])
                labels.extend([l.item() for l in batch.labels])
                ids.extend([i.item() for i in batch.image_number])
        
        return metrics(labels, preds), ids


    def batch_accuracy(self, predictions, labels):
        """
        Calculates mean batch accuracy for multiclass data
        
        predictions: model output after forward pass
        
        labels: list of one-hot encoded labels
        """
        
        max_predictions = predictions.argmax(dim=1, keepdim=True)
        correct = max_predictions.squeeze(1).eq(labels)
        
        return correct.sum() / torch.FloatTensor([labels.shape[0]])


    def binary_accuracy(self, predictions, labels):
        """
        Calculates mean batch accuracy for binary data

        predictions: model output after forward pass
          
        labels: list of one-hot encoded labels
        """
        rounded_preds = torch.round(self.sig(predictions))
        
        correct = (rounded_preds == labels).float() 
        acc = correct.sum() / len(correct)
        
        return acc

    
    

"""## Run Model"""

#define hyperparameters
INPUT_DIM = len(TEXT.vocab)
PAD_ID = TEXT.vocab.stoi[TEXT.pad_token]
UNK_ID = TEXT.vocab.stoi[TEXT.unk_token]
N_EPOCHS = 50
EARLY_STOPPING = {"patience": 10, "delta": 0.01}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1, 1]
OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text"


results_dict = {}
max_f1_value = 0

for i in LEARNING_RATES:
    # Initialize LSTM model
    model = LSTMClassifier(train_set, dev_set, test_set, PAD_ID, INPUT_DIM)  

    # Load pretrained vector
    model.embedding.weight.data.copy_(TEXT.vocab.vectors) 

    # Manually initialize UNK and PAD tokens as zero vectors (and NOT randomly as would be done otherwise)
    model.embedding.weight.data[UNK_ID] = torch.zeros(300)
    model.embedding.weight.data[PAD_ID] = torch.zeros(300)

    #model test
    train_dict = model.trainer(model, i, EARLY_STOPPING, N_EPOCHS) 
    (results_dict[i], labels, preds), ids = model.test(model)

    if results_dict[i]["f1"] >= max_f1_value: #only save best model
        max_f1_value = results_dict[i]["f1"]
        model_saver(model, "LSTM", OUTPUT_DIR, train_dict, labels, preds, results_dict[i])

        
# save complete training results
np.save(os.path.join(OUTPUT_DIR, "dogwhistle_total_training_results.npy"), results_dict)
df = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/LSTM/LSTM_dogwhistle_predictions.csv")
df["image_number"] = ids
df.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/LSTM/LSTM_dogwhistle_predictions.csv")




X_TICK_LABELS = ["None", "Hateful", "Counter-speech", "Reclaimed"]
Y_TICK_LABELS = ["None", "Hateful", "Counter-speech", "Reclaimed"]
COLOR = "Blues"
SAVE_NAME = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/LSTM/LSTM_cm_dogwhistle.png"
CM = np.load("/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/LSTM/LSTM_dogwhistle_test_results.npy", allow_pickle=True)


confusion_matrix_plotter(CM, SAVE_NAME, X_TICK_LABELS, Y_TICK_LABELS, COLOR)

