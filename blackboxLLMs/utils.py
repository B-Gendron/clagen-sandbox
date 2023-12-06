import numpy as np
import datasets
from math import sqrt
from datasets import load_dataset
import torch
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import string
from sklearn.metrics import f1_score, classification_report, confusion_matrix, matthews_corrcoef, precision_score, recall_score
from termcolor import colored

dailydialog = load_dataset('daily_dialog')

def format_prompt_from_dialog(split, dial_id, utt_id):
    '''
        This auxiliary function concats the dialog utterances in a string by adding hyphens and line breaks at each utterance separation.

        @param dialog (list): a dialog represented as a list of utterances

        @return one_line (str): a string containing the dialog content and some formatting characters
    '''
    dialog = dailydialog[split][dial_id]['dialog']
    utterance = dialog[utt_id]
    header = "Here is a dialog: \n"
    one_line = '\n- ' + '\n-'.join(dialog)
    footer = f"\n\nRegarding its conversational context, give the appropriate emotion to describe this utterance: '{utterance}', amongst: happiness, sadness, anger, surprise, fear, disgust. If none of them seems to correspond, the appropriate answer is no emotion. In this case, the most appropriate emotion label is:"
    return header + one_line + footer

def format_prompt_last_utterance(split, dial_id):
    '''
        This auxiliary function concats the dialog utterances in a string by adding hyphens and line breaks at each utterance separation.

        @param dialog (list): a dialog represented as a list of utterances

        @return one_line (str): a string containing the dialog content and some formatting characters
    '''
    dialog = dailydialog[split][dial_id]['dialog']
    header = "Here is a dialog: \n"
    one_line = '\n- ' + '\n-'.join(dialog)
    footer = f"\n\nRegarding its conversational context, give the appropriate emotion for the last utterance among: happiness, sadness, anger, surprise, fear, disgust. If none of them seems to correspond, the appropriate answer is no emotion. In this case, the most appropriate emotion label is:"
    return header + one_line + footer

def format_prompt_last_utterance_falcon(split, dial_id):
    '''
        This auxiliary function concats the dialog utterances in a string by adding hyphens and line breaks at each utterance separation.

        @param dialog (list): a dialog represented as a list of utterances

        @return one_line (str): a string containing the dialog content and some formatting characters
    '''
    dialog = dailydialog[split][dial_id]['dialog']
    header = "Here is a dialog: \n"
    one_line = '\n- ' + '\n-'.join(dialog)
    footer = f"\n\nRegarding its conversational context, return the appropriate emotion for the last utterance among: sadness, happiness, anger, surprise, fear and disgust. If none of them properly correspond, return 'no emotion'."
    return header + one_line + footer

# TEST SAMPLE
# processed_sample = format_prompt_from_dialog('train', 0, 0)

def minimum_index(l):
    min_index = 0
    list_len = len(l)
    for index in range(list_len):
        if l[index] < l[min_index]:
            min_index = index
    return min_index


def predicted_emotion(output):
    '''
        Use the word_in_string auxiliary function to find the predicted emotion in Llama output.
    '''
    emotions = ['happiness', 'fear', 'anger', 'disgust', 'surprise', 'sadness', 'no emotion']
    indexes_list = [len(output)+1 for _ in range(len(emotions))]
    truncated_output = output[output.find("is:"):]
    splitted_output = truncated_output.translate(str.maketrans('', '', string.punctuation)).split()
    new_output = ' '.join(splitted_output)
    for i in range(len(emotions)):
        word = emotions[i]
        find_index = new_output.find(word)
        if find_index > -1:
            indexes_list[i] = find_index
    return emotions[minimum_index(indexes_list)]

# TEST SAMPLE
# output = "Regarding its conversational context, give the appropriate emotion to describe this utterance: 'blablabla', amongst: happiness, sadness, anger, surprise, fear, disgust, no emotion. In this case, the most appropriate emotion label is: happiness.\nReason: The utterance is a proposal for an activity that is typically enjoyed as a form of recreation or relaxation. The tone is friendly and casual, suggesting a relaxed and lighthearted atmosphere. There is no evidence of sadness, anger, or fear in the text, and the utterance does not convey a sense of disgust or surprise. Therefore, the appropriate emotion label is happiness."
# print(predicted_emotion(output))

def map_emotion_to_index(emotion):
    '''
        An auxiliary function that converts an emotion label to its associated index according to dailydialog construction.

        @param emotion (str): the emotion label. Must be one of these: no emotion, happiness, anger, sadness, surprise, fear, disgust.

        @returns index (int): an integer between 0 and 6 that represent the emotion label in dailydialog.
    '''
    labels = {'no emotion':"0", 'anger':"1", 'disgust':"2", 'fear':"3", 'happiness':"4", 'sadness':"5", 'surprise':"6"}
    return int(labels[emotion])


def custom_flatten(ll):
    '''
        A function to flatten a list of lists where sub lists are of heterogeneous sizes.

        @param ll (list): the input list of lists

        @return l (list): the flattened list   
    '''
    l = []
    for sl in ll:
        l.extend(sl)
    return l


def compute_metrics_and_variance(all_trues, all_preds):
    mcc_torch_list, mcc_list, prec_list, rec_list, f1_list = [], [], [], [], []

    for i in range(len(all_trues)):
        trues, preds = all_trues[i], all_preds[i]

        # compute classification metrics on each run
        metric = MulticlassMatthewsCorrCoef(num_classes=7)
        mcc_torch = metric(torch.Tensor(preds), torch.Tensor(trues))
        mcc = matthews_corrcoef(trues, preds)
        prec = precision_score(trues, preds, average='weighted', zero_division=0)
        rec = recall_score(trues, preds, average='weighted')
        f1 = f1_score(trues, preds, average='weighted')
        cm = confusion_matrix(trues, preds)

        # store values in corresponding lists
        mcc_torch_list.append(mcc_torch.item())
        mcc_list.append(mcc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
    
    std_mcc_torch, mean_mcc_torch = np.std(mcc_torch_list), np.mean(mcc_torch_list)
    std_mcc, mean_mcc = np.std(mcc_list), np.mean(mcc_list)
    std_prec, mean_prec  = np.std(prec_list), np.mean(prec_list)
    std_rec, mean_rec = np.std(rec_list), np.mean(rec_list)
    std_f1, mean_f1  = np.std(f1_list), np.mean(f1_list)

    results = {'mean' : [mean_mcc_torch, mean_mcc, mean_prec, mean_rec, mean_f1], 'std' : [std_mcc_torch, std_mcc, std_prec, std_rec, std_f1], 'cm': cm}

    return results


def store_classification_metrics(results, model):
    means = results['mean']
    std = results['std']

    with open(f"falcon-7b-last.txt", "a") as f:
        print(f"CLASSIFICATION SCORES FOR {model} ON DAILYDIALOG TEST SET",file=f)
        print("", file=f)
        print("Confusion matrix", results['cm'], file=f)
        print("", file=f)
        print('MCC (Torch): ', means[0], '+/-', std[0], file=f)
        print('MCC (Sklearn): ', means[1], '+/-', std[1], file=f)
        print('Precision: ', means[2], '+/-', std[2], file=f)
        print('Recall: ', means[3], '+/-', std[3], file=f)
        print('Weighted F1 score: ', means[4], '+/-', std[4], file=f)

def compute_mcc_from_cm(TP, TN, FP, FN):
    top = TP*TN - FP*FN
    bottom = (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)
    return top / sqrt(bottom)