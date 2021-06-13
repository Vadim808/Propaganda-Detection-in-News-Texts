from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.debug = True
app.config['CORS_HEADERS'] = 'Content-Type'

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

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def preprocessing(text, max_len=300):
    col_input_ids = []
    col_attention_mask = []
    col_token_type_ids = []
    technique = []
    res_texts = []
  	
    token_text = tokenizer.encode_plus(
        text, 
        return_offsets_mapping=True,
        max_length=max_len,
        truncation=True
    )
    
    token_count = len(token_text.input_ids)
    
    for label in labels:
        token_technique = tokenizer.encode_plus(
            label, 
            return_offsets_mapping=True, 
            max_length=max_len, 
            truncation=True
        )
        input_ids = token_text.input_ids + token_technique.input_ids[1:]
        token_type_ids = [0] * token_count + [1] * len(token_technique.input_ids[1:])
        len_input_ids = len(input_ids)
        attention_mask = [1] * len_input_ids
        
        if max_len < len_input_ids:
            break

        technique.append(label)
        res_texts.append(text)
        padding = [0] * (max_len - len_input_ids)
        col_input_ids.append(input_ids + padding)
        col_attention_mask.append(attention_mask + padding)
        col_token_type_ids.append(token_type_ids + padding)

    return col_input_ids, col_attention_mask, col_token_type_ids, technique, res_texts


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 


class Model(transformers.BertPreTrainedModel):
    def __init__(self, config, PATH):
        super(Model, self).__init__(config)
        self.bert = transformers.BertModel.from_pretrained(PATH)
        self.linear = nn.Linear(768, 1)
        self.flatten = nn.Flatten()
        self.sigm = nn.Sigmoid()
    
    def forward(self, ids, mask, token_type_ids):
        embedding = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )[0]
        logits = self.linear(embedding)
        logits = self.flatten(logits)
        result = self.sigm(logits)
        return result

PATH = "DeepPavlov/rubert-base-cased"

bert = transformers.BertModel.from_pretrained(PATH)
model = Model(bert.config, PATH)


PATH = "propaganda-model/dict_mymodel_best_2.pth"
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))


@app.route("/", methods=['GET', 'POST'])
def index():
    EMB_SIZE = 300
    text = request.form["name"]
    col_input_ids, col_attention_mask, col_token_type_ids, technique, result_text = preprocessing(text, max_len=300)
    ids = torch.tensor(col_input_ids).to(dev)
    attention_mask = torch.tensor(col_attention_mask).to(dev)
    type_ids = torch.tensor(col_token_type_ids).to(dev)
    result_labels = [0] * EMB_SIZE
    result_prop = [0] * EMB_SIZE
    model.eval()
    output = model(ids, attention_mask, type_ids)
    for i in range(len(labels)):
	    for j in range(EMB_SIZE):
	    	if result_prop[j] < output[i][j]:
	    		result_prop[j] = output[i][j]
	    		result_labels[j] = i
    res_text = ""
    ans_seq_tok = []
    PROP_LINE = 0.95
    flag = 0
    for i in range(1, EMB_SIZE):
        if col_input_ids[0][i] == 102:
            break

        if i > 0:
            diff = result_labels[i] != result_labels[i - 1]
        else:
            dif = True

        if flag == 1 and (result_prop[i] <= PROP_LINE or diff == True):
            new_text = tokenizer.decode(ans_seq_tok)
            res_text += new_text + "</mark> "
            ans_seq_tok = []
            flag = 0

        if flag == 0 and result_prop[i] > PROP_LINE:
            if len(ans_seq_tok) != 0:
                new_text = tokenizer.decode(ans_seq_tok)
            else:
            	new_text = ""
            color = "style='border: 1px dashed'"
            res_text += new_text + f" <mark title='{labels[result_labels[i]]}' {color}>"
            ans_seq_tok = []

        flag = result_prop[i] > PROP_LINE
        ans_seq_tok.append(col_input_ids[0][i])

    new_text = tokenizer.decode(ans_seq_tok)
    if flag == 1:
        res_text += new_text + "</mark> "
    else:
        res_text += new_text

    res_text = res_text.replace(' ##', '')
    print(res_text, "success")

    return res_text

if __name__ == "__main__":
    app.run()
