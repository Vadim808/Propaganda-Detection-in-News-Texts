from os import listdir
import pandas as pd
import numpy as np
import json

TEXTS_PATH1 = "data2/datasets1/train-articles"
LABELS_PATH1 = "data2/datasets1/train-labels-task2-technique-classification"

TEXTS_PATH2 = "data2/datasets2/train-articles"
LABELS_PATH2 = "data2/datasets2/train-labels-task2-technique-classification"

TEXTS_NAMES1 = listdir(TEXTS_PATH1)
TEXTS_NAMES2 = listdir(TEXTS_PATH2)
PART_TEXT_NAMES = np.concatenate([TEXTS_NAMES1, TEXTS_NAMES2])

LABELS_NAMES = []
TEXTS_NAMES = []
for i, name in enumerate(PART_TEXT_NAMES): 
    if i < len(TEXTS_NAMES1):
        TEXTS_NAMES += [TEXTS_PATH1 + "/" + name]
        LABELS_NAMES += [LABELS_PATH1 + "/" + name[:-3] + "task2-TC.labels"]
    else:
        TEXTS_NAMES += [TEXTS_PATH2 + "/" + name]
        LABELS_NAMES += [LABELS_PATH2 + "/" + name[:-3] + "task2-TC.labels"]
        
def read_labels_file(label_path):
    f = open(label_path, "r")
    labels = []
    for string in f:
        num, technique, start, end = string.split("\t")
        labels += [[int(start), int(end[:-1]), technique]]
    labels.sort()
    return labels

def read_text_file(text_path):
    f = open(text_path, "r")
    return f.read()

def separate_text(text, labels):
    sep_data = []
    positions = [0]
    texts = text.split("\n")
    ind = 0
    for i in range(len(texts)):
        ind += len(texts[i]) + 1
        positions += [ind]
    c_pos = 0
    for label in labels:
        while (positions[c_pos] <= label[0] < positions[c_pos + 1]) == 0:
            c_pos += 1
            if c_pos + 1 == len(positions):
                return sep_data
        left_edge = positions[c_pos]
        sep_data += [[texts[c_pos], 
                      label[0] - left_edge,
                      label[1] - left_edge,
                      label[2]]]
    return sep_data

def add_json_type(d):
    el = [{
            "start": d[1],
            "end": d[2],
            "technique": d[3]
         }]
    return el

labels = []
texts = []
data = []
for i in range(len(TEXTS_NAMES)):
    text_path = TEXTS_NAMES[i]
    label_path = LABELS_NAMES[i]
    labels = read_labels_file(label_path)
    text = read_text_file(text_path)
    sep_data = separate_text(text, labels)
    if len(sep_data) == 0:
        continue
    c_text = sep_data[0][0]
    label = []
    for d in sep_data:
        if d[0] == c_text:
            label += add_json_type(d)
        else:
            if len(label) != 0:
                data += [[c_text, label]]
            c_text = d[0]
            label = add_json_type(d)
    data += [[c_text, label]]
    
data_pd = pd.DataFrame(data, columns=["text", "labels"])

labels = [
    "Reductio ad hitlerum",
    "Whataboutism",
    "Presenting Irrelevant Data (Red Herring)",
    "Doubt",
    "Slogans",
    "Appeal to fear/prejudice",
    "Obfuscation, Intentional vagueness, Confusion",
    "Misrepresentation of Someone's Position (Straw Man)",
    "Glittering generalities (Virtue)",
    "Appeal to authority",
    "Repetition",
    "Bandwagon",
    "Causal Oversimplification",
    "Name calling/Labeling",
    "Thought-terminating cliché",
    "Flag-waving",
    "Exaggeration/Minimisation",
    "Smears",
    "Loaded Language",
    "Black-and-white Fallacy/Dictatorship"
]

new_corpus_labels = [
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Exaggeration,Minimisation",
    "Doubt",
    "Appeal_to_fear-prejudice",
    "Flag-Waving",
    "Causal_Oversimplification",
    "Slogans",
    "Appeal_to_Authority",
    "Black-and-White_Fallacy",
    "Thought-terminating_Cliches",
    "Whataboutism",
    "Reductio_ad_Hitlerum",
    "Red_Herring",
    "Bandwagon",
    "Obfuscation,Intentional_Vagueness,Confusion",
    "Straw_Men"
]

new_corpus_labels.sort()
labels.sort()
labels.remove("Glittering generalities (Virtue)")
labels.remove("Misrepresentation of Someone's Position (Straw Man)")

error = f"labels_len {len(labels)} != len_new_corpus_labels {len(new_corpus_labels)}"
assert len(labels) == len(new_corpus_labels), error

d_change_labels = dict()
for i in range(len(labels)):
    d_change_labels[new_corpus_labels[i]] = labels[i]
d_change_labels["Straw_Men"] = "Misrepresentation of Someone's Position (Straw Man)"

def correct_technique(old_t, name, data, i):
    old_t["technique"] = name
    data["labels"][i].append(old_t)
    
wrong_name1 = ["Bandwagon", "Reductio_ad_Hitlerum"]
wrong_name2 = ["Whataboutism", "Straw_Men", "Red_Herring"]

def change_technique_name(data, d_change_labels):
    for i in range(data.shape[0]):
        N = len(data["labels"][i])
        j = 0
        while j < N:
            old_name = data["labels"][i][j]["technique"]
            if "Bandwagon,Reductio_ad_hitlerum" == old_name:
                old_t = data["labels"][i].pop(j)
                for name in wrong_name1:
                    correct_technique(old_t.copy(), name, data, i)
                continue
            if "Whataboutism,Straw_Men,Red_Herring" == old_name:
                old_t = data["labels"][i].pop(j)
                for name in wrong_name2:
                    correct_technique(old_t.copy(), name, data, i)
                continue
            new_name = d_change_labels[data["labels"][i][j]["technique"]]
            data["labels"][i][j]["technique"] = new_name
            j += 1
            
change_technique_name(data_pd, d_change_labels)
data_pd.to_csv("data2/datasets1/new_data2.csv", index=False)

train_data1_pd = pd.read_json("data1/training_set_task2.txt")
train_data2_pd = data_pd.copy()
train_data1_pd = train_data1_pd.drop(columns=["id"])
train_pd = pd.concat([train_data1_pd, train_data2_pd], ignore_index=True)

def separation_of_topics(data):
    texts = []
    prop_mask = []
    techniques = []
    for i in range(data.shape[0]):
        text = data["text"][i]
        dict_prop = dict()
        for label in data["labels"][i]:
            if label == '[':
                print(data["labels"][i], len(data["labels"][i]))
            technique = label["technique"]
            if technique not in dict_prop:
                dict_prop[technique] = np.array([0] * len(text))
            dict_prop[technique][label["start"]:label["end"]] = 1
        #for technique in labels:
        #    if technique not in dict_prop:
        #        dict_prop[technique] = np.array([0] * len(text))
        for key in dict_prop:
            texts.append(text)
            techniques.append(key)
            prop_mask.append(dict_prop[key])
    return texts, prop_mask, techniques

texts, prop_mask, techniques = separation_of_topics(train_pd)
sep_train_pd = pd.DataFrame(np.array([texts, prop_mask, techniques], dtype=object).T,
                            columns = ["text", "prop_mask", "technique"])

sep_train_pd.to_csv("data.csv", index=False)