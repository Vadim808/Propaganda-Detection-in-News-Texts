import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def hist_graph(model, i, line, sep_dev_pd, tokenizer, dev="cpu"):
    #take data
    ids = torch.tensor([list(sep_dev_pd["col_input_ids"][i])]).to(dev)
    attention_mask = torch.tensor([list(sep_dev_pd["col_attention_mask"][i])]).to(dev)
    type_ids = torch.tensor([list(sep_dev_pd["col_token_type_ids"][i])]).to(dev)
    #take model
    model.eval()
    with torch.no_grad():
        ans_seq_tok = np.array(sep_dev_pd["col_input_ids"][i])
        output = model(ids, attention_mask, type_ids)
        ans_mask = (torch.squeeze(output, dim=1)[0]).cpu()
        plt.figure(figsize=(18,8))
        words = []
        end = 0
        for j in range(len(ans_seq_tok)):
            if ans_seq_tok[j] == 102:
                end = j
                break
            words += [tokenizer.decode(int(ans_seq_tok[j])) + "_" + str(j)]
        plt.hlines(line, xmin=0, xmax=end, color ='r')
        plt.plot(words, ans_mask[:end], drawstyle='steps-mid', label='steps-mid', color ='black')
    plt.show()

def text_check(model, i, line, sep_dev_pd, tokenizer, dev="cpu"):
    #take data
    ids = torch.tensor([list(sep_dev_pd["col_input_ids"][i])]).to(dev)
    attention_mask = torch.tensor([list(sep_dev_pd["col_attention_mask"][i])]).to(dev)
    type_ids = torch.tensor([list(sep_dev_pd["col_token_type_ids"][i])]).to(dev)
    #take model
    model.eval()
    with torch.no_grad():
        output = model(ids, attention_mask, type_ids)
        ans_mask = (torch.squeeze(output, dim=1)[0] > line).cpu()
        print("technique:", sep_dev_pd["technique"][i])
        print("---")
        print("text:", sep_dev_pd["text"][i])
        print("---")
        ans_seq_tok = np.array(sep_dev_pd["col_input_ids"][i])[ans_mask == 1]
        print("ans:", tokenizer.decode(ans_seq_tok))
        print("---")
        ans_mask = np.array(sep_dev_pd["col_token_prop_mask"][i])
        ans_seq_tok = np.array(sep_dev_pd["col_input_ids"][i])[ans_mask == 1]
        print("true ans:", tokenizer.decode(ans_seq_tok))
        print("---")
        hist_graph(model, i, line, sep_dev_pd, tokenizer)
        plt.show()

def check_metrics(model, data, line=0.5, dev="cpu"):
    batch_size = 2
    M = data.shape[0]
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, M, batch_size):
            start = i
            end = i + batch_size if i + batch_size < M else M
            ids = torch.tensor(list(data["col_input_ids"][start:end])).to(dev)
            attention_mask = torch.tensor(list(data["col_attention_mask"][start:end])).to(dev)
            type_ids = torch.tensor(list(data["col_token_type_ids"][start:end])).to(dev)
            target = torch.tensor(list(data["col_token_prop_mask"][start:end]), dtype=torch.float).to(dev)
            
            output = model(ids, attention_mask, type_ids)
            output = (torch.squeeze(output, dim=1) > line).type(torch.int16)
            sum_tens = output + target
            diff_tens = output - target
            tp += (sum_tens == 2).sum().item()
            fp += (diff_tens == 1).sum().item()
            tn += (sum_tens == 0).sum().item()
            fn += (diff_tens == -1).sum().item()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    F = 2 * precision * recall / (precision + recall)
    print(f"precision={precision}, \nrecall={recall}, \naccuracy={accuracy}, \nF={F}")
    return precision, recall, accuracy, F