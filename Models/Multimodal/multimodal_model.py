# -*- coding: utf-8 -*-
"""Multimodal_model.ipynb

This script trains a multimodal model feature concatenation model over a hyperparameter sweep and evaluates its performance. 

"""

import pandas as pd
import os
import numpy as np
import torch
import random
import functools
import operator
import cv2
import seaborn as sns
import matplotlib.pyplot as plt


from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer, AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.models import load_model, Model


%cd "/content/drive/My Drive/Dog_Whistle_Code"
from HelperFunctions import *

"""### Helper Functions"""

def model_saver(model, model_type, output_directory, training_dict, labels, preds, results, tokenizer= None):
    """
    Saves PyTorch Model and other outputs

    model: Model to be saved
    
    model_type (string): Name of model
    
    output_directory: Directory to folder to save file in

    training_dict: Dictionary of training and validation values 

    labels: List of labels for test set

    preds: List of model predictions after passed through argmax()

    results: Dictionary of metrics

    tokenizer: Tokenizer to be saved. Defaulted to None.
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    os.chdir(output_directory)

    np.save(model_type+"_dogwhistle_train_results.npy", training_dict) #save training dict
    np.save(model_type+"_dogwhistle_test_results.npy", results) #save test metrics
    
    test_predictions = pd.DataFrame([labels, preds]) #save predictions and labels 
    test_predictions = test_predictions.T
    test_predictions = test_predictions.rename(columns={0: 'Labels', 1: 'Predictions'})
    test_predictions.to_csv(model_type+"_dogwhistle_predictions.csv")

    #save models
    torch.save(model.state_dict(), model_type+"_model")

    return print("Saving complete.")

"""### Feature Extractors"""

class Transformer_features(nn.Module):
    def __init__(self):
        def __init__(self, method_type):
        """
        method_type: Extracts features from Bert either using the method in Devlin et al or Sabat el al

        """
        super(Transformer_features, self).__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if method_type == "Devlin":
                self.config = AutoConfig.from_pretrained('/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/RoBERTa', output_hidden_states = True)
                self.model = AutoModel.from_config(self.config).to(self.device)

            if method_type == "Sabat":
                self.model = RobertaModel.from_pretrained('/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/RoBERTa').to(self.device)


    def forward(self, dataloader, method_type):
        """
        This function recieves tokenized tensors and the sentence pair IDs and returns a sentence embedding for each input sequence

        dataloader: dataloader object containing combined text and IDs

        method_type: Extracts features from Bert either using the method in Devlin et al or Sabat el al

        """
   
        self.model.eval()


        if method_type == "Devlin": # averages word embeddings to get sentence embeddings, then concatenates last four layers

            combined_layers = torch.zeros(1, 4096).to(self.device)
            id_list = []

            for batch in dataloader:
                with torch.no_grad():
                    _, _, encoded_layers = self.model(batch[0].to(self.device), attention_mask=batch[1].to(self.device)) #shape [25 x len(tokens) x 100 x 1024]

                concat_layers = torch.cat((torch.mean(encoded_layers[-4], dim=1), torch.mean(encoded_layers[-3], dim=1), torch.mean(encoded_layers[-2], dim=1), torch.mean(encoded_layers[-1], dim=1)), dim=1)
                combined_layers = torch.cat((combined_layers, concat_layers), dim=0)
                id_list.append(batch[2])

        if method_type == "Sabat": # averages word embeddings from last layer 

            combined_layers = torch.zeros(1, 1024).to(self.device)
            id_list = []

            for batch in dataloader:
                with torch.no_grad():
                    output, _ = self.model(batch[0].to(self.device)) #shape [batch_size x pad_length x 1024]

                text_features = torch.mean(output, dim=1)
                combined_layers = torch.cat((combined_layers, text_features), dim=0)
                id_list.append(batch[2])

        
        combined_layers = combined_layers[1:, :].to(torch.int64) #input len x 4096
        id_list = torch.as_tensor(functools.reduce(operator.iconcat, id_list, [])).to(torch.int64) #input length
        out_matrix = torch.cat((id_list.unsqueeze(dim= 1).to(self.device), combined_layers.to(self.device)), dim=1)

        return out_matrix


"""### Model Setup"""

class MultimodalClassifier(nn.Module):
    def __init__(self, MLP_type, image_model, text_model, hidden_size: int=100, dropout: float=0.2, num_labels: int=4, input_len: int):
        """Initializes the network structure
        
        MLP_type: Which paper's MLP structure to use

        image_model: CovNet from Keras library to use as image feature extractor

        text_model: Transformer Model from Hugging Face to use as the text feature extractor

        hidden_size (int): Number of nodes in the hidden layer. Defaulted to 100 

        dropout (float): Rate at which nodes are deactivated. Defaulted to 0.2. 
        
        num_labels (int): Number of labels to predict. Defaulted to 4.

        input_len (int): Length of input vector. 
        """
        
        super(MultimodalClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_model = image_model
        self.text_model = text_model


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



    def forward(self, images, text_tokens, text_masks):
        """Initiaties foward pass through network
        
        images: batch of image arrays from dataloader

        text_tokens: batch of RoBERTa text token arrays from dataloder

        text_masks: batch of text mask arrays from dataloader
        """
        
        batch_size = images.size()[0] 

        #Initialize input feature vector
        features = torch.zeros(batch_size, 1).to(self.device)

        #Extract image features
        image_features = Model(self.image_model.input, self.image_model.layers[-2].output).predict(images.cpu().numpy())
        #image_features = self.image_model.extract_features(images.cpu().numpy())
        image_features_torch = torch.Tensor(image_features)
        features = torch.cat((features, image_features_torch.to(self.device)), dim=1)

        #Extract text feature      
        text_features = self.text_model(text_tokens, text_masks)
        features = torch.cat((features, text_features), dim=1)

        features = features[:, 1:] #remove extra zero from initialization

        out = self.classifier(features.to(self.device))

        return out


    def trainer(self, model, train, dev, early_stop_vals: dict, epochs: int = 25, learning_rate: float = 1e-5, weight_decay: float = 0.1, warmup: float = 0.06):   
            """
            Trains multimodal model

            model: Instatiation of model

            train: Dataloader object containing train data- image, text, labels

            dev: Dataloader object containing dev data- image, text, labels

            early_stopping: Dictionary containing patience value (int) and delta value (float). The patience determines the number of epochs to wait to achieve the given delta

            epochs (int): Number of times to run through all batches. Default value is 25.

            learning_rate (float): Default value is 1e-5.

            weight decay (float): Default value is 0.1 

            warmup (float): Default value is 0.06; percentage of training steps in warmup.
            """

            self.optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay) 
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = warmup * (len(train) * epochs), num_training_steps = (1-warmup) * (len(train) * epochs))

            train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []

            for epoch in trange(epochs, desc= "Epoch"):
                if early_stopping(val_loss_values, early_stop_vals) == False:
                    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
                    print('Training...')

                    train_total_loss, train_total_len, train_num_correct = 0, 0, 0

                    model.train()

                    for step, batch in enumerate(train): 
                        if step % 50 == 0:
                            print("Processing batch...{} of {}".format(step, len(train)))

                        model.zero_grad()  
                        batch_images, batch_text_tokens, batch_text_masks, batch_labels = tuple(t.to(self.device) for t in batch)
                        train_total_len += batch_images.shape[0]

                        predictions = model(batch_images, batch_text_tokens, batch_text_masks)

                        criterion = nn.CrossEntropyLoss().to(self.device)
                        loss = criterion(predictions, batch_labels) 
                        train_total_loss += loss

                        loss.backward() # Backpropagate loss
                        self.optimizer.step() # Update weights
                        self.scheduler.step()

                        pred = predictions.argmax(1, keepdim=True).float()
                        correct_tensor = pred.eq(batch_labels.float().view_as(pred))
                        correct = np.squeeze(correct_tensor.cpu().numpy())
                        train_num_correct += np.sum(correct)

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


                    train_acc = train_num_correct / train_total_len
                    train_acc_values.append(train_acc)
                    avg_train_loss = train_total_loss / len(train)         
                    train_loss_values.append(avg_train_loss)

                    print()
                    print("Running Validation...")
                    print()

                    val_total_loss, val_total_len, val_num_correct = 0, 0, 0 

                    model.eval()

                    for batch in dev:
                        batch_images, batch_text_tokens, batch_text_masks, batch_labels = tuple(t.to(self.device) for t in batch)
                        val_total_len += batch_images.shape[0]

                        with torch.no_grad():        

                            predictions = model(batch_images, batch_text_tokens, batch_text_masks)
      
                        loss = criterion(predictions, batch_labels) 
                        val_total_loss += loss
                      
                        pred = predictions.argmax(1, keepdim=True).float()
                        correct_tensor = pred.eq(batch_labels.float().view_as(pred))
                        correct = np.squeeze(correct_tensor.cpu().numpy())
                        val_num_correct += np.sum(correct)

                    val_acc = val_num_correct / val_total_len
                    val_acc_values.append(val_acc)
                    avg_val_loss = val_total_loss / len(dev)
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


    def test(self, model, test):
        """
        Tests the model's performance based on a several metrics

        model: Instatiation of model

        test: Dataloader object containing test data- image, text, labels
        """
   
        print('Predicting labels for {} sentences...'.format(len(test)))
        
        model.eval()

        predictions, true_labels = [], []

        for batch in test:
            batch_images, batch_text_tokens, batch_text_masks, batch_labels = tuple(t.to(self.device) for t in batch)
      
            with torch.no_grad():
                preds = model(batch_images, batch_text_tokens, batch_text_masks)

            predictions.append(preds.detach().cpu().numpy())
            true_labels.append(batch_labels.to('cpu').numpy())

        predictions = functools.reduce(operator.iconcat, predictions, [])
        true_labels = functools.reduce(operator.iconcat, true_labels, [])

        print('    DONE.')
    
        return metrics(true_labels, predictions, argmax_needed= True)

    
    
#Define hyperparameters
NORMALIZE_LIST = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']
ANNOTATE_LIST = ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']
TOKENIZER = AutoTokenizer.from_pretrained('/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/RoBERTa')
DROPOUT = 0.2
HIDDEN_SIZE = 100
BATCH_SIZE = 32
NUM_LABELS = 4
NUM_EPOCHS = 30 #50
EARLY_STOPPING = {"patience": 30, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.1 
WARMUP = 0.06 
OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Multimodal/Feature Concatenation"
INPUT_LEN = 2048


#Instantiate Feature Extractors
ImageExtractor = load_model('/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Image/Xception/image_model.h5') #using pre-trained Xception

TextExtractor = Transformer_features() #using pre-trained RoBERTa

"""### Data Preparation"""

class DogWhistleDataset(Dataset):
    def __init__(self, df, base_path, tokenizer, image_size: int=299, pad_length: int=100):
        self.data = df
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.pad_length = pad_length
         
    def __len__(self):
        return (self.data.shape[0])
    
    def __getitem__(self, i):
        image_path = str(self.data.loc[i, "image_number"])
        path = self.base_path + "/" + image_path + ".jpg"
        image = cv2.imread(path) 
        image = cv2.resize(image, (self.image_size, self.image_size)) 

        text = self.data.loc[i, "combined_text"] 
        encoded_dict = self.tokenizer.encode_plus(text, add_special_tokens = True, max_length = self.pad_length, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

        label = self.data.loc[i, "Primary_numeric_gt"] #torch.Tensor(self.data.loc[i, "Primary_numeric_gt"]).to(torch.int64)

        sample = (image, torch.sum(encoded_dict['input_ids'], dim=0), torch.sum(encoded_dict['attention_mask'], dim=0), label) #reshape encoded_dict from 1x100 to 100

        return sample

#Load data
train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/dog_whistle_train.csv", encoding='utf-8')
dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/dog_whistle_dev.csv", encoding='utf-8')
test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/dog_whistle_test.csv", encoding='utf-8')


#Clean data
train["combined_text"] = combine_text(train)
train["combined_text"] = clean_text(train["combined_text"], NORMALIZE_LIST, ANNOTATE_LIST)
dev["combined_text"] = combine_text(dev)
dev["combined_text"] = clean_text(dev["combined_text"], NORMALIZE_LIST, ANNOTATE_LIST)
test["combined_text"] = combine_text(test)
test["combined_text"] = clean_text(test["combined_text"], NORMALIZE_LIST, ANNOTATE_LIST)


#Subset necessary data
train = train[["image_number", "combined_text", "Primary_numeric_gt"]]
dev = dev[["image_number", "combined_text", "Primary_numeric_gt"]] 
test = test[["image_number", "combined_text", "Primary_numeric_gt"]] 


#Create Dataset
train_dataset = DogWhistleDataset(train, "/content/drive/My Drive/Dog_Whistle_Code/Data/Images", TOKENIZER)
dev_dataset = DogWhistleDataset(dev, "/content/drive/My Drive/Dog_Whistle_Code/Data/Images", TOKENIZER)
test_dataset = DogWhistleDataset(test, "/content/drive/My Drive/Dog_Whistle_Code/Data/Images", TOKENIZER)


#Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



"""### Run Model"""

Classifier = MultimodalClassifier(ImageExtractor, TextExtractor, HIDDEN_SIZE, DROPOUT, NUM_LABELS, INPUT_LEN)
train_dict = Classifier.trainer("Sabat", Classifier, train_dataloader, dev_dataloader, EARLY_STOPPING, NUM_EPOCHS, 0.1, WEIGHT_DECAY, WARMUP)
metrics, labels, preds = Classifier.test(Classifier, test_dataloader)
model_saver(Classifier, "Multimodal", OUTPUT_DIR, train_dict, labels, preds, metrics)

print(metrics)

#Hyperparameter search

results_dict = {}
for i in LEARNING_RATES: 
    Classifier = MultimodalClassifier( ) #Add inputs 
    train_dict = Classifier.train(train_dataloader, dev_dataloader, PAD_LENGTH, EARLY_STOPPING, NUM_EPOCHS, i, WEIGHT_DECAY, WARMUP, INPUT_LEN)
    results_dict[j], labels, preds = Classifier.test(test_dataloader) #Update inputs

    if results_dict[j]["f1"] >= max_f1_value: #only save best model
        max_f1_value = results_dict[j]["f1"]
        print("The new top F1 score is: {}. Saving model...".format(max_f1_value))
        model_saver(Classifier, "Multimodal", OUTPUT_DIR, train_dict, labels, preds, results_dict[j])


#save complete training results
np.save(os.path.join(os.path.join(OUTPUT_DIR, "Multimodal"), "dogwhistle_mm_total_training_results.npy"), results_dict)

