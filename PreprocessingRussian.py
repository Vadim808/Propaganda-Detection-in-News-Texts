import pandas as pd

data_pd = pd.read_csv("data.csv")
data_pd = data_pd.reset_index().drop(columns=["index"])

def make_list(x):
    x = x[1:-1].split()
    x = list(map(int, x))
    return x

data_pd["prop_mask"] = data_pd["prop_mask"].apply(lambda x: make_list(x))

def preprocessing_translate(text, mask):
    text_parts = []
    text_part = ""
    c_flag = mask[0]
    parts_type = [c_flag] 
    for i, flag in enumerate(mask):
        if c_flag == flag:
            text_part += text[i]
        else:
            c_flag = flag
            parts_type += [c_flag]
            text_parts += [text_part]
            text_part = text[i]
    if text_part == "":
        parts_type.pop(-1)
    else:
        text_parts += [text_part]
    return text_parts, parts_type

import string
import re

def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text

from argostranslate import package, translate
package.install_from_path('translate-en_ru-1_1.argosmodel')
installed_languages = translate.get_installed_languages()
translation_en_rus = installed_languages[0].get_translation(installed_languages[1])

def preproccesing_connect(text_parts, parts_type):
    prop_mask = []
    text = []
    for i, text_part in enumerate(text_parts):
        text_part = normalize_text(text_part)
        rus_text = translation_en_rus.translate(text_part)
        text_len = len(rus_text)
        part_type = parts_type[i]
        if part_type == 1 or (part_type == 0 and len(text_parts) == 1):
            prop_mask += [part_type] * text_len
        elif i == 0 or i == len(text_parts) - 1:
            prop_mask += [part_type] * (text_len + 1)
        else:
            prop_mask += [part_type] * (text_len + 2)
        text += [rus_text]
    return prop_mask, " ".join(text)

import numpy as np

new_rus_texts = []
new_prop_mask = []
for i in range(data_pd.shape[0]):
    texts, part_types = preprocessing_translate(data_pd["text"][i], data_pd["prop_mask"][i])
    prop_mask, text = preproccesing_connect(texts, part_types)
    new_rus_texts += [text]
    new_prop_mask += [prop_mask]
    if i % 100 == 0:
	    print(i, data_pd.shape[0])

techniques = data_pd["technique"]
new_data = pd.DataFrame(np.array([new_rus_texts, new_prop_mask, techniques], dtype=object).T,
                        columns = ["text", "prop_mask", "technique"])

new_data.to_csv("rus_data.csv", index=False)