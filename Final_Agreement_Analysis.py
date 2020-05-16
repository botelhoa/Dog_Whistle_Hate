#!/usr/bin/env python
# coding: utf-8

"""
Script to compute Kappa scores of annotation rounds, identify areas of disagreement
where expert decision is needed, and to collate final ground truth dataset
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import statistics
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
pd.options.display.max_colwidth = 500
os.chdir("D:\\MMHS150K\\Dog_Whistle6k\\Annotations")


def merge_annotations(file_path1, file_path2):
    """ Merges annotations from two annotators to a single DataFrame
    file_path1: directory to the file containing annotator 1's spreadsheet
    
    file_path2: directory to the file containing annotator 2's spreadsheet
    """
    
    labeled_data = {}
    
    annotator1_df = pd.read_excel(file_path1).fillna("none")
    annotator2_df = pd.read_excel(file_path2).fillna("none")
    
    for i in range(len(annotator1_df)):
        if annotator1_df.loc[i, "Tweet ID"] == annotator2_df.loc[i, "Tweet ID"]:
            tweet_dict = {}
            labeled_data[annotator1_df.loc[i, "Tweet ID"]] = tweet_dict
            for column in annotator1_df.columns[:7]:
                tweet_dict[column] = [annotator1_df.loc[i, column].lower(), annotator2_df.loc[i, column].lower()]
                
        else:
            print("Issue with row {}".format(i))
    
    df = pd.DataFrame(labeled_data)
    df = df.T

    for i in ["Links"]:
        del df[i]

    tweet_ids = []

    for i in df["Tweet ID"]:
        split = i[0].split("'")
        if len(split) == 2:
            tweet_ids.append(split[1])
        else:
            tweet_ids.append(split)
            
    df["Tweet ID"] = tweet_ids 
    
    return df.reset_index(drop=True)



def make_numeric(data):
    """ Converts string annotations into numerics
    data: dataframe with string annotations
    """
    df = data.copy()
    
    LABEL_DICTIONARY= { 
        "Primary": {"none": 0, "hateful": 1, "counter-speech": 2, "reclaimed": 3},
        "Modality": {"none": 0, "text-unimodal": 1, "image-unimodal": 2, "multimodal": 3},
        "Strength": {"none": 0, "animosity": 1, "derogation": 2, "extreme": 3}
                    } 
    
    for column in LABEL_DICTIONARY.keys():
        annotator_1 = df[column].map(lambda x: x[0])
        annotator_2 = df[column].map(lambda x: x[1])
        annotator_1 = annotator_1.map(LABEL_DICTIONARY[column])
        annotator_2 = annotator_2.map(LABEL_DICTIONARY[column])
        
        for i in range(len(df)):
            df.loc[i, column] = [annotator_1[i], annotator_2[i]]
        
    
    df["Strategies"] = df["Strategies"].map(lambda x: strategies(x))
            
    return df


def strategies(x):
    """ Returns numeric values for Strategy column annotations
    x: List of values (length 2)
    """
    new_vals = [0, 0]
    
    for i in range(len(x)):
        if x[i] == "none":
            new_vals[i] = 0
        else:
            if "explicit" in x[i]:
                new_vals[i] = 1
            else:
                new_vals[i] = 2

    return new_vals
    

def expert_needed(data, columns):
    """ Flags values where ground truth is ambigious
    data: dataframe
    
    columns: list of column names where ground truth is needed
    """
    
    df = data.copy()
    
    
    for i in columns:
        ground_truth_list = []
        for j in range(len(df)):
            row_val = df.loc[j, i]
            if row_val[0] != row_val[1]:
                ground_truth_list.append(1)
            else:
                ground_truth_list.append(0)
            
        df["Expert Needed- {}".format(i)] = ground_truth_list
    
    return df
    
    
def annotator_agreement(data):
    '''
    Function to return number of annotators in agreement
    
    data: dataframe with annotator labels saved as list in each cell
    '''
    
    data1 = data.copy()
    
    column_names = [i for i in data.columns if i != "target"]
    
    kappas = {}
    
    for column in column_names:
        
        annotator_1 = list(data1[column].map(lambda x: x[0]))
        annotator_2 = list(data1[column].map(lambda x: x[1]))

        kappas[column] = cohen_kappa_score(annotator_1, annotator_2)

    return  kappas


def target():
    """ TODO: distill targets to groups
    """
    
    return


# ### Kappas and Expert Decisions per Batch
# Compute Kappa score for annotations for each batch. Identify disagreements to add to spreadsheet for expert decision.


#df_merged = merge_annotations(" ", " ")
#df_merged = merge_annotations(" ", " ")
#df_merged = merge_annotations(" ", " ")
#df_merged = merge_annotations(" ", " ")
#df_merged = merge_annotations(" ", " ")
df_merged = merge_annotations(" ", " ")
df_numeric = make_numeric(df_merged)
df_expert = expert_needed(df_numeric, columns=["Primary", "Modality", "Strategies"])
kappas = annotator_agreement(df_numeric[["Primary", "Modality", "Strength", "Strategies"]])


#Check if any data is missing
df_numeric.isna().sum()

#Inspect kappa scores
kappas


#Insepect number of expert decisions by columns
print(df_expert["Expert Needed- Primary"].sum())
print(df_expert["Expert Needed- Modality"].sum())
print(df_expert["Expert Needed- Strategies"].sum())


#Return IDs needing Expert decision

expert_tweet_ids = []

for i in ["Expert Needed- Primary", "Expert Needed- Modality", "Expert Needed- Strategies"]:
    df_temp = df_expert[df_expert[i]==1]
    temp_expert_ids = df_temp["Tweet ID"].values.tolist()
    expert_tweet_ids.extend(temp_expert_ids)
    
expert_tweet_ids = set(expert_tweet_ids)
    
print(len(expert_tweet_ids))
print()

for i in expert_tweet_ids:
    print("'"+i) #add string to make transfer to excel easier


#Return annotator labels

for col in ["Primary", "Modality"]: #, "Strategies"]:
    print("Listing values for {}".format(col))
    for i in expert_tweet_ids: #print in same order as above
        temp_df = df_merged[df_merged["Tweet ID"] == i]
        temp_idx = temp_df.index[0]
        print(temp_df.loc[temp_idx, col])
    print("--------------------------------------------")
    
#handling for strategies columns which will be made binary for expert

strategy_to_string_dict = {0: "none", 1: "explicit", 2: "dog-whistle"}

print("Listing values for strategies")
for i in expert_tweet_ids: #print in same order as above
    temp_df = df_numeric[df_numeric["Tweet ID"] == i]
    temp_idx = temp_df.index[0]
    temp_list = temp_df.loc[temp_idx, "Strategies"]
    print([strategy_to_string_dict[temp_list[0]], strategy_to_string_dict[temp_list[1]]])


# ### Collate Final Dataset
# Merge annotations and expert decisions across rounds into final dataset containg string and numeric versions of labels. Add ground truth and compute expert agreement metric.


def merge_numeric_string(df_string, df_numeric):
    """Merges datasets and renames columns
    
    df_string: dataframe containing labels as strings
    
    df_numeric: dataframe containing labels as numerics
    """
    temp_df = df_string.merge(df_numeric, on="Tweet ID")
    temp_df = temp_df.rename(columns={'Modality_x': 'Modality_string', 'Primary_x': 'Primary_string', 'Strategies_x': 'Strategies_string', 'Strength_x': 'Strength_string', 'Target_x': 'Target_string',
                     'Modality_y': 'Modality_numeric', 'Primary_y': 'Primary_numeric', 'Strategies_y': 'Strategies_numeric', 'Strength_y': 'Strength_numeric'})
    temp_df = temp_df.drop(columns= "Target_y")
   
    return temp_df


def expert_loader(file_path, row_num):
    """Loads spreadhseet with expert rulings, fills na, and adds a column of numeric values
    
    file_path: directory to file
    
    row_num: row number where annotator pairs switch in expert spreadsheet
    """
    
    LABEL_DICTIONARY= { 
        "Primary": {"none": 0, "hateful": 1, "counter-speech": 2, "reclaimed": 3},
        "Modality": {"none": 0, "text-unimodal": 1, "image-unimodal": 2, "multimodal": 3},
        "Strategies": {"none": 0, "explicit": 1, "dog-whistle": 2}
                    } 
   
    df_expert = pd.read_excel(file_path)
    df_expert = df_expert.fillna("none")
    df_expert["Tweet ID"] = df_expert["Tweet ID"].map(lambda x: tweet_id_helper(x))
    
    for column in ["Primary", "Modality", "Strategies"]:
        df_expert[column] = df_expert[column].map(lambda x: x.lower())
    
    df_expert_numeric = df_expert.copy()
    
    for column in LABEL_DICTIONARY.keys():
        df_expert_numeric[column] = df_expert_numeric[column].map(LABEL_DICTIONARY[column])
        
    df_expert = df_expert.merge(df_expert_numeric, on="Tweet ID")
    df_expert = df_expert.rename(columns={'Modality_x': 'Modality_string', 'Primary_x': 'Primary_string', 'Strategies_x': 'Strategies_string',
                     'Modality_y': 'Modality_numeric', 'Primary_y': 'Primary_numeric', 'Strategies_y': 'Strategies_numeric'})
    
    
    #Calculate agreement level between expert and annotators
    df_expert_slice1 = df_expert.loc[:row_num]
    df_expert_slice2 = df_expert.loc[row_num:]
    annotator1_dict, annotator2_dict = expert_agreement(df_expert_slice1, ["Primary", "Modality", "Strategies"])
    annotator3_dict, annotator4_dict = expert_agreement(df_expert_slice2, ["Primary", "Modality", "Strategies"])
    annotator_dict = {"Annotator1": annotator1_dict, "Annotator2": annotator2_dict, "Annotator3": annotator3_dict, "Annotator4": annotator4_dict}
    
    df_expert = df_expert[['Tweet ID','Modality_string', 'Primary_string', 'Strategies_string', 'Modality_numeric', 'Primary_numeric','Strategies_numeric']]
    
    return df_expert, annotator_dict


def tweet_id_helper(tweet_id):
    """Helper function since some tweet id's start with " ' " and others do not
    
    tweet_id: tweet_id in string format
    """
    if len(tweet_id) == 20:
        tweet_id = tweet_id.split("'")
        return tweet_id[1]
    else:
        return tweet_id

    
def ground_truth_dataset(original_data1, original_data2, expert_data):
    """Returns the dataset
    original data: dataframe with original annotations
    
    expert_data: dataframe with expert decision on disagreement
    """
    
    GT_COLUMNS = ["Primary_string", "Modality_string", "Primary_numeric", "Modality_numeric", "Strategies_numeric"]
    
    df_gt = pd.concat([original_data1, original_data2]).reset_index(drop=True)
    
#     for column in GT_COLUMNS:
#         temp_column = []
#         expert_column = []
#         for idx, row_val in enumerate(df_gt[column]):
#             if row_val[0] == row_val[1]:
#                 temp_column.append(row_val[0])
#                 expert_column.append(np.nan)
#             else:
#                 tweet_id = df_gt.loc[idx, "Tweet ID"]
#                 temp_expert = expert_data[expert_data["Tweet ID"] == tweet_id]
#                 temp_column.append(temp_expert[column].values[0])
#                 expert_column.append(temp_expert[column].values[0])
        
#         df_gt["{}_gt".format(column)] = temp_column
#         df_gt["{}_expert".format(column)] = expert_column

    for column in GT_COLUMNS:
        temp_column = []
        expert_column = []
        for idx, row_val in enumerate(df_gt[column]):
            tweet_id = df_gt.loc[idx, "Tweet ID"]
            if tweet_id in expert_data["Tweet ID"].values.tolist(): #this handling allows for expert to override original labels
                temp_expert = expert_data[expert_data["Tweet ID"] == tweet_id]
                temp_column.append(temp_expert[column].values[0])
                expert_column.append(temp_expert[column].values[0])
            else:
                temp_column.append(row_val[0])
                expert_column.append(np.nan)
        
        df_gt["{}_gt".format(column)] = temp_column
        df_gt["{}_expert".format(column)] = expert_column

    
    return df_gt


def unimodal_converter(data):
    """
    Returns ground truth labels when data is split modally to text and image
    
    data: dataframe object
    """
    
    for column in ["string", "numeric"]:

        unimodal_image, unimodal_text = [], []
    
        for i in range(len(data)):
            temp_val = data.loc[i, "Modality_{}_gt".format(column)]
            if temp_val in ["text-unimodal", "image-unimodal", 1, 2]: #prevelance of image-only hate low enough to treat image and text unimodal equally because of image text
                if column == "string":
                    unimodal_image.append("none")
                if column == "numeric":
                    unimodal_image.append(0)
                unimodal_text.append(data.loc[i, "Primary_{}_gt".format(column)])
            if temp_val in ["multimodal", 3]:
                unimodal_image.append(data.loc[i, "Primary_{}_gt".format(column)])
                unimodal_text.append(data.loc[i, "Primary_{}_gt".format(column)])

            if temp_val in ["none", 0]:
                if column == "string":
                    unimodal_image.append("none")
                    unimodal_text.append("none")
                if column == "numeric":
                    unimodal_image.append(0)
                    unimodal_text.append(0)
        
        data["Unimodal_text_{}".format(column)] = unimodal_text
        data["Unimodal_image_{}".format(column)] = unimodal_image
    
    return data


def expert_agreement(data, columns):
    """
    Returns percentage agreement between annotator and expert per column when there is a disagreement between annotators
    
    data: dataframe object of ground truth dataset
    
    columns: list of columns to compare annotator performance with expert decision
    """
    
    annotator1_dict = {}
    annotator2_dict = {}
    
    for column in columns:
        annotator1 = list(data["{}- Original Annotations_x".format(column)].map(lambda x: string_list_converter(x)[0]))
        annotator2 = list(data["{}- Original Annotations_x".format(column)].map(lambda x: string_list_converter(x)[1]))
        expert_rulings = list(data["{}_string".format(column)])
        
        temp_df = pd.concat([pd.DataFrame(annotator1, columns = ["Annotator_1"]), pd.DataFrame(annotator2, columns= ["Annotator_2"]), pd.DataFrame(expert_rulings, columns= ["Expert"])], axis=1)
        
        annotator1_agreement = 0
        annotator2_agreement = 0 
        total_disagreement = 0
        
        for i in range(len(temp_df)):
            if temp_df.loc[i, "Annotator_1"] != temp_df.loc[i, "Annotator_2"]:
                total_disagreement += 1
                if temp_df.loc[i, "Annotator_1"] == temp_df.loc[i, "Expert"]:
                    annotator1_agreement += 1
                if temp_df.loc[i, "Annotator_2"] == temp_df.loc[i, "Expert"]:
                    annotator2_agreement += 1
        
        annotator1_dict[column] = annotator1_agreement/total_disagreement      
        annotator2_dict[column] = annotator2_agreement/total_disagreement 
        
    return annotator1_dict, annotator2_dict


def string_list_converter(string):
    """ Converts lists that are strings ie "['element1', 'element2']" to list
    string: string element
    """
    new_list = []
    new_string = string.split(",")
    
    new_list.append(new_string[0].split("[")[1].strip("'"))
    new_list.append(new_string[1].strip().split("]")[0].strip("'"))
    
    return new_list

def is_ambiguous(data, column_list):
    """ Returns a boolean list for whether there was annotator disagreement
    data: DataFrame object
    
    column_list: list of column names
    """
    
    for column in column_list:
        is_ambiguous = []
        for row in data[column]:
            if row in [0.0, 1.0, 2.0, 3.0]:
                is_ambiguous.append(1)
            else:
                is_ambiguous.append(0)
    
        data["{}_is_ambiguous".format(column)] = is_ambiguous
        
    return data
    
    
def strength_ambiguous(series):
    """Returns binary column of whether there was annotator disagreement for the Strength
    
    series: iterable object with rows containing list of annotator labels
    """
    
    is_ambiguous = []
    
    for i in series:
        annotation1 = i[0]
        annotation2 = i[1]
        
        if annotation1 == annotation2:
            is_ambiguous.append(1)
        else:
            is_ambiguous.append(0)
            
    return is_ambiguous


#Create final dataset

#Load annotation sheets
df1 = merge_annotations(" ", " ")
df2 = merge_annotations(" ", " ")
df3 = merge_annotations(" ", " ")
df4 = merge_annotations(" ", " ")
df5 = merge_annotations(" ", " ")
df6 = merge_annotations(" ", " ")


#load expert ruling sheets
df1_expert, annotator_dict1 = expert_loader(" ", 245)
df2_expert, annotator_dict2 = expert_loader(" ", 501)
df3_expert, annotator_dict3  = expert_loader(" ", 478)

#Convert to numeric- annotation sheets
df1_numeric = make_numeric(df1)
df2_numeric = make_numeric(df2)
df3_numeric = make_numeric(df3)
df4_numeric = make_numeric(df4)
df5_numeric = make_numeric(df5)
df6_numeric = make_numeric(df6)


#Merge
df1_numeric_string = merge_numeric_string(df1, df1_numeric)
df2_numeric_string = merge_numeric_string(df2, df2_numeric)
df3_numeric_string = merge_numeric_string(df3, df3_numeric)
df4_numeric_string = merge_numeric_string(df4, df4_numeric)
df5_numeric_string = merge_numeric_string(df5, df5_numeric)
df6_numeric_string = merge_numeric_string(df6, df6_numeric)


#Add Ground Truth
df1_gt = ground_truth_dataset(df1_numeric_string, df2_numeric_string, df1_expert)
df2_gt = ground_truth_dataset(df3_numeric_string, df4_numeric_string, df2_expert)
df3_gt = ground_truth_dataset(df5_numeric_string, df6_numeric_string, df3_expert)


#combine all and final touches
df_final = pd.concat([df1_gt, df2_gt]).reset_index(drop=True) #df3_gt
df_final = unimodal_converter(df_final)
df_final = is_ambiguous(df_final, ["Primary_numeric_expert", "Modality_numeric_expert", "Strategies_numeric_expert"])
df_final["Strength_numeric_is_ambiguous"] = strength_ambiguous(df_final["Strength_numeric"])
df_final.to_pickle("final_ground_truth.pkl")



#Create dictionary of kappa scores
kappa_dict = {}

kappa_dict["Batch 1"] = annotator_agreement(df1_numeric[["Primary", "Modality", "Strength", "Strategies"]])
kappa_dict["Batch 2"] = annotator_agreement(df2_numeric[["Primary", "Modality", "Strength", "Strategies"]])
kappa_dict["Batch 3"] = annotator_agreement(df3_numeric[["Primary", "Modality", "Strength", "Strategies"]])
kappa_dict["Batch 4"] = annotator_agreement(df4_numeric[["Primary", "Modality", "Strength", "Strategies"]])
kappa_dict["Batch 5"] = annotator_agreement(df5_numeric[["Primary", "Modality", "Strength", "Strategies"]])
kappa_dict["Batch 6"]= annotator_agreement(df6_numeric[["Primary", "Modality", "Strength", "Strategies"]])

batches = ["Batch 1", "Batch 2", "Batch 3", "Batch 4", "Batch 5", "Batch 6"]
column_names = ["Primary", "Modality", "Strength", "Strategies"]
kappa_lists = []

for i in column_names:
    col_list = []
    for j in batches:
        col_list.append(kappa_dict[j][i])
        
    kappa_lists.append(col_list)


#Overall kappa score
average_kappa = .2*((kappa_lists[0][0] +kappa_lists[0][1])/2) + .4*((kappa_lists[0][2] + kappa_lists[0][3])/2) + .4*((kappa_lists[0][4] + kappa_lists[0][5])/2) #weighted average because first 2 batches had fewer tweets
print("Average Primary Kappa Score Across Batches: {}".format(average_kappa))


#Visualize kappa scores by column over time
sns.set(style='darkgrid')
plt.rcParams["figure.figsize"] = (12,6)

plt.plot([((kappa_lists[0][0]+kappa_lists[0][1])/2), ((kappa_lists[0][2]+kappa_lists[0][3])/2), ((kappa_lists[0][4]+kappa_lists[0][5])/2)], label="Primary")
plt.plot([((kappa_lists[1][0]+kappa_lists[1][1])/2), ((kappa_lists[1][2]+kappa_lists[1][3])/2), ((kappa_lists[1][4]+kappa_lists[1][5])/2)], label="Modality")
plt.plot([((kappa_lists[2][0]+kappa_lists[2][1])/2), ((kappa_lists[2][2]+kappa_lists[2][3])/2), ((kappa_lists[2][4]+kappa_lists[2][5])/2)], label="Strength")
plt.plot([((kappa_lists[3][0]+kappa_lists[3][1])/2), ((kappa_lists[3][2]+kappa_lists[3][3])/2), ((kappa_lists[3][4]+kappa_lists[3][5])/2)], label="Strategies")

plt.xlabel("Rounds")
plt.xticks([0, 1, 2], labels= ["1", "2", "3"])
plt.ylabel("Kappa Score")
plt.legend()

plt.savefig("Kappa_scores_across_rounds.png",bbox_inches='tight')
plt.show()



#Expert agreement by annotator
annotator1 = {"Round 1": annotator_dict1["Annotator1"], "Round 2": annotator_dict2["Annotator2"] "Round 3": annotator_dict3["Annotator1"]}
annotator2 = {"Round 1": annotator_dict1["Annotator4"], "Round 2": annotator_dict2["Annotator1"] "Round 3": annotator_dict3["Annotator3"]}
annotator3 = {"Round 1": annotator_dict1["Annotator2"], "Round 2": annotator_dict2["Annotator4"] "Round 3": annotator_dict3["Annotator4"]}
annotator4 = {"Round 1": annotator_dict1["Annotator3"], "Round 2": annotator_dict2["Annotator3"] "Round 3": annotator_dict3["Annotator2"]}



#Note: expert can disagree with both annotators, this the numbers to not add up to 2
agreement_percents = []
for num, i in enumerate([annotator1, annotator2, annotator3, annotator4]):
    annotator_vals = []
    for j in ["Round 1", "Round 2"]:#, "Round 3"]:
        annotator_vals.append(i[j]["Primary"])
        annotator_vals.append(i[j]["Modality"])
        annotator_vals.append(i[j]["Strategies"])
        
    agreement_percents.append(np.mean(annotator_vals))
    print("Annotator {}'s average expert agreement: {}".format(num+1, np.mean(annotator_vals)))
        
print("Expert agreed with no one: ", 2-np.sum(agreement_percents))

