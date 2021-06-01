# -*- coding: utf-8 -*-
"""Multimodal Feature Extraction

This script separately extracts image and text features using fine-tuned RoBERTa and Xception models and saves them to a .csv file

"""


import pandas as pd
import os
import numpy as np
import torch
import functools
import operator
import cv2


from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, AutoTokenizer, AutoModel, AutoConfig
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.models import load_model, Model


"""### Text Feature Extraction"""

class Transformer_features(nn.Module):
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

    
#Text Hyperparameters
NORMALIZE_LIST = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']
ANNOTATE_LIST = ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']
TOKENIZER = RobertaTokenizer.from_pretrained('/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Text/RoBERTa')
PAD_LENGTH = 100

class DogWhistleDatasetText(Dataset):
    def __init__(self, df, tokenizer, pad_length: int=100):
        self.data = df
        self.tokenizer = tokenizer
        self.pad_length = pad_length
         
    def __len__(self):
        return (self.data.shape[0])
    
    def __getitem__(self, i):
        text = self.data.loc[i, "combined_text"] 
        encoded_dict = self.tokenizer.encode_plus(text, add_special_tokens = True, max_length = self.pad_length, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

        image_number = self.data.loc[i, "image_number"]

        return (torch.sum(encoded_dict['input_ids'], dim=0), torch.sum(encoded_dict['attention_mask'], dim=0), image_number) #reshape encoded_dict from 1x100 to 100

# Prepare data

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
train = train[["image_number", "combined_text"]]
dev = dev[["image_number", "combined_text"]] 
test = test[["image_number", "combined_text"]] 


#Create Dataset
train_dataset = DogWhistleDatasetText(train, TOKENIZER)
dev_dataset = DogWhistleDatasetText(dev, TOKENIZER)
test_dataset = DogWhistleDatasetText(test, TOKENIZER)


#Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32)
dev_dataloader = DataLoader(dev_dataset, batch_size=32) 
test_dataloader = DataLoader(test_dataset, batch_size=32)

TextExtractor = Transformer_features("Devlin")

train_text_features = TextExtractor(train_dataloader, "Devlin")
print("Done")
dev_text_features = TextExtractor(dev_dataloader, "Devlin")
print("Done")
test_text_features = TextExtractor(test_dataloader, "Devlin")
print("Done")

TextExtractor = Transformer_features("Sabat")

train_text_features = TextExtractor(train_dataloader, "Sabat")
print("Done")
dev_text_features = TextExtractor(dev_dataloader, "Sabat")
print("Done")
test_text_features = TextExtractor(test_dataloader, "Sabat")
print("Done")

#Save Devlin
train_text_features = train_text_features.cpu().numpy()
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/text_features.npy", train_text_features)

dev_text_features = dev_text_features.cpu().numpy()
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/text_features.npy", dev_text_features)

test_text_features = test_text_features.cpu().numpy()
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/text_features.npy", test_text_features)

#Save Sabat
train_text_features= train_text_features.cpu().numpy()
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/text_features_sabat.npy", train_text_features)

dev_text_features = dev_text_features.cpu().numpy()
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/text_features_sabat.npy", dev_text_features)

test_text_features = test_text_features.cpu().numpy()
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/text_features_sabat.npy", test_text_features)

"""### Image Feature Extraction"""

def Image_features(trained_model, dataloader):
    """ Extracts image features from images

    trained_model: pre-trained image model

    dataloader: dataloader object containing image paths and IDs
    """

    combined_output = np.zeros((1, 1024))
    id_list = []

    for num, batch in enumerate(dataloader):
        if num % 25 == 0:
            print("Processing batch {} of {}".format(num, len(dataloader)))
        batch_output = Model(trained_model.input, trained_model.layers[-2].output).predict(batch[0]) #32 x 1024
        combined_output = np.concatenate((combined_output, batch_output), axis=0)
        id_list.append(batch[1])


    combined_output = combined_output[1:, :]
    id_list = np.array(functools.reduce(operator.iconcat, id_list, []))

    out_matrix = np.concatenate((np.expand_dims(id_list, axis=1), combined_output), axis=1)

    return out_matrix



class DogWhistleDatasetImage(Dataset):
    def __init__(self, df, base_path, image_size: int=299):
        self.data = df
        self.base_path = base_path
        self.image_size = image_size

    def __len__(self):
        return (self.data.shape[0])
    
    def __getitem__(self, i):
        image_path = str(self.data.loc[i, "image_number"])
        path = self.base_path + "/" + image_path + ".jpg"
        image = cv2.imread(path) 
        image = cv2.resize(image, (self.image_size, self.image_size)) 

        sample = (image, self.data.loc[i, "image_number"])

        return sample
    
    

#Load data
train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/dog_whistle_train.csv", encoding='utf-8')
dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/dog_whistle_dev.csv", encoding='utf-8')
test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/dog_whistle_test.csv", encoding='utf-8')


#Subset necessary data
train = train[["image_number"]]
dev = dev[["image_number"]] 
test = test[["image_number"]] 


#Create Dataset
train_dataset = DogWhistleDatasetImage(train, "/content/drive/My Drive/Dog_Whistle_Code/Data/Images")
dev_dataset = DogWhistleDatasetImage(dev, "/content/drive/My Drive/Dog_Whistle_Code/Data/Images")
test_dataset = DogWhistleDatasetImage(test, "/content/drive/My Drive/Dog_Whistle_Code/Data/Images")


#Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32)
dev_dataloader = DataLoader(dev_dataset, batch_size=32) 
test_dataloader = DataLoader(test_dataset, batch_size=32)

ImageExtractor = load_model('/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Image/Xception/image_model.h5') #using pre-trained Xception

train_image_features = Image_features(ImageExtractor, train_dataloader)
dev_image_features = Image_features(ImageExtractor, dev_dataloader)
test_image_features = Image_features(ImageExtractor, test_dataloader)

# Save
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/image_features.npy", train_image_features)
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/image_features.npy", dev_image_features)
np.save("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/image_features.npy", test_image_features)



"""### Combine Feature Data"""

# Load Text data
# train_text = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/text_features.npy", allow_pickle=True)
# dev_text = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/text_features.npy", allow_pickle=True)
# test_text = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/text_features.npy", allow_pickle=True)
train_text = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/text_features_sabat.npy", allow_pickle=True)
dev_text = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/text_features_sabat.npy", allow_pickle=True)
test_text = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/text_features_sabat.npy", allow_pickle=True)


# Load Image data
train_image = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/image_features.npy", allow_pickle=True)
dev_image = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/image_features.npy", allow_pickle=True)
test_image = np.load("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/image_features.npy", allow_pickle=True)

# Load Other data
train2 = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/dog_whistle_train.csv", encoding='utf-8')
dev2 = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/dog_whistle_dev.csv", encoding='utf-8')
test2 = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/dog_whistle_test.csv", encoding='utf-8')

# Merge
train = pd.concat((pd.DataFrame(train_text[:, 1:]), pd.DataFrame(train_image[:, 1:])), axis = 1)
train["ids"] = train_text[:, :1]
train["labels"] = train2["Primary_numeric_gt"]
dev = pd.concat((pd.DataFrame(dev_text[:, 1:]), pd.DataFrame(dev_image[:, 1:])), axis = 1)
dev["ids"] = dev_text[:, :1]
dev["labels"] = dev2["Primary_numeric_gt"]
test = pd.concat((pd.DataFrame(test_text[:, 1:]), pd.DataFrame(test_image[:, 1:])), axis = 1)
test["ids"] = test_text[:, :1]
test["labels"] = test2["Primary_numeric_gt"]

# Save
# train.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/combined_features.csv")
# dev.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/combined_features.csv")
# test.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/combined_features.csv")
train.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/combined_features_sabat.csv")
dev.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/combined_features_sabat.csv")
test.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/combined_features_sabat.csv")






