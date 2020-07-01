# -*- coding: utf-8 -*-
"""MMBT.ipynb

This py script trains and tests a Multi-Modal Bit Transformer from the Simple Transformers Library
"""

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from simpletransformers.classification.multi_modal_classification_model import MultiModalClassificationModel
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

#Create DataFrame with three columns text, image, labels

def combine_text(df):
    """
    Combines tweet and image text into one column

    df: Dataframe which holds the data
    """
    combined_text = []

    for row_num in range(len(df)):
        tweet_text = df.loc[row_num, "tweet_text"]
        image_text = df.loc[row_num, "img_text"]
        if type(image_text) == str:
            combined_text.append(tweet_text + image_text)
        else:
            combined_text.append(tweet_text)

    return combined_text

def clean_text(data, normalize_list, annotate_list):
    """
    This function preprocesses the text using the Ekphrasis library
    
    data: Pandas series object containing strings of text

    normalize_list: list of data features to clean

    annotate_list: list of data features to annotate
    """

    text_processor = TextPreProcessor(
        normalize= normalize_list,
        annotate= annotate_list,
        fix_html=True,
        segmenter="twitter", 
        unpack_hashtags=True,  
        unpack_contractions=True,  
        spell_correct_elong=True,  
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    )

    clean_data = data.map(lambda x: " ".join(text_processor.pre_process_doc(x)))

    return clean_data


def df_preparer(file_path):
    """
    file_path: 
    """

    df = pd.read_csv(file_path, encoding='utf-8')
    df["text"] = combine_text(df)
    df["text"] = clean_text(df["text"], ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'], ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'])
    df["labels"] = df["Primary_numeric_gt"].map(lambda x: str(x)) #labels need to be strings
    df["images"] = df["image_number"].map(lambda x: str(x) + ".jpg")

    df = df[["labels", "text", "images"]]

    return df


def metrics(labels, preds, argmax_needed: bool = False):
    """
    Returns the Matthew's correlation coefficient, accuracy rate, true positive rate, true negative rate, false positive rate, false negative rate, precission, recall, and f1 score
    
    labels: list of correct labels

    pred: list of model predictions

    argmax_needed (boolean): converts logits to predictions. Defaulted to false.
    """
    labels = labels
    preds = preds

    if argmax_needed == True:
        preds = np.argmax(preds, axis=1).flatten()

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    f1 = f1_score(labels, preds, average= "weighted")
    precision = precision_score(labels, preds, average= "weighted")
    recall = f1_score(labels, preds, average= "weighted")

    results = {
        "mcc": mcc,
        "acc": acc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return results, labels, preds

# Prepare data
train = df_preparer("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/dog_whistle_train.csv")
dev = df_preparer("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/dog_whistle_dev.csv")
test = df_preparer("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/dog_whistle_test.csv")
test["int_labels"] = test["labels"].map(lambda x: int(x))



# Run Model
args = {"fp16": False, 'train_batch_size': 8, 'eval_batch_size': 8, 'learning_rate': 1e-5, 'num_train_epochs': 4, 'max_seq_length': 100, "weight_decay": 0.1, "warmup_ratio": 0.06,
        "use_early_stopping": True, "early_stopping_patience": 2, "early_stopping_delta": 0, "early_stopping_metric": "eval_loss", "overwrite_output_dir": True}

model = MultiModalClassificationModel("bert", "bert-large-uncased", label_list= ["0", "1", "2", "3"], args=args, use_cuda= True)
print("Loading finished")
model.train_model(train, eval_data=dev, image_path="/content/drive/My Drive/Dog_Whistle_Code/Data/Images/", output_dir= "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Multimodal/MMBT")
print("Training finished")
results, logits  = model.eval_model(test, image_path="/content/drive/My Drive/Dog_Whistle_Code/Data/Images/")
metric_vals, labels, preds = metrics(test["int_labels"], logits, argmax_needed=True)
print(metric_vals)


#Save
np.save("/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Multimodal/MMBT/results.npy", metric_vals)

df = pd.DataFrame([labels, preds]).T
df = df.rename(columns ={"int_labels": "Labels", "Unnamed 0": "Predictions"})
df.to_csv("/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Multimodal/MMBT/predictions.csv")

