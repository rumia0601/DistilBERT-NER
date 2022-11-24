#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
from transformers import DistilBertForTokenClassification
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
import re


# In[2]:


def align_word_ids(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


# In[3]:


def evaluate_one_text(model, sentence):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    unique_labels2 =set()
    unique_labels2.add('I-TR')
    unique_labels2.add('I-PR')
    unique_labels2.add('B-TR')
    unique_labels2.add('I-TE')
    unique_labels2.add('B-PR')
    unique_labels2.add('O')
    unique_labels2.add('B-TE')
    print(unique_labels2)
    ids_to_labels2 = {v: k for v, k in enumerate(sorted(unique_labels2))}
    print(ids_to_labels2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    #print(len(text))

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]
    
    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels2[i] for i in predictions]
    return prediction_label


# In[4]:


class DistilbertNER(nn.Module):
  def __init__(self, tokens_dim):
    super(DistilbertNER,self).__init__()
    if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')
    if tokens_dim <= 0:
          raise ValueError('Classification layer dimension should be at least 1')
    self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = tokens_dim) #set the output of each token classifier = unique_lables

  def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss

    if labels == None:
      out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
      return out

    out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
    return out


# In[5]:


class NLPmodule:
    input_val=[]
    #def __init__(self,input_val):
    #    self.input_val = []

    def append_element(idx, tag1, PR2, TE2, TR2):
    
        if(tag1 == 'B-PR' or tag1 == 'I-PR'):
            PR2.append(idx)
        elif(tag1 == 'B-TE' or tag1 == 'I-TE'):
            TE2.append(idx)
        elif(tag1 == 'B-TR' or tag1 == 'I-TR'):
            TR2.append(idx)

    def GetNerToken(self, target):
        model = torch.load("./NER_model" , "cpu")

        #input_text = input()

        #text = input_text
        #special_characters = '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]'
        #for character in special_characters:
        #    text = text.replace(character, " " + character + " ") 
        text = ' '.join(target)
        text = re.sub(r"\s+", " ", text)
        #pre-processing
        if (text[0] == " "):
            text = text[1:]
        label = evaluate_one_text(model, text)
        for i in range(len(label)):
            if (i == 0) and (label[i] == 'I-PR'):
                label[i] = 'B-PR'
            elif (i == 0) and (label[i] == 'I-TE'):
                label[i] = 'B-TE'
            elif (i == 0) and (label[i] == 'I-TR'):
                label[i] = 'B-TR'
            elif (i != 0) and (label[i] == 'I-PR') and ((label[i - 1] != 'B-PR' and label[i - 1] != 'I-PR')):
                label[i] = 'B-PR'
            elif (i != 0) and (label[i] == 'I-TE') and ((label[i - 1] != 'B-TE' and label[i - 1] != 'I-TE')):
                label[i] = 'B-TE'
            elif (i != 0) and (label[i] == 'I-TR') and ((label[i - 1] != 'B-TR' and label[i - 1] != 'I-TR')):
                label[i] = 'B-TR'
#post-processing
        
        print(text)
        print(label)
        pattern = "[ ]"
        words = re.split(pattern, text)
        for i in range(len(words)):
            words[i] += "="
        maps = list(map(str.__add__, words, label))
        print(maps)

#change output format
        tagging = []
        PR = []
        TE = []
        TR = []
        PR2 = []
        TE2 = []
        TR2 = []

        for idx in range(len(maps)):
            if(idx < len(maps)-1):
                word1, tag1 = maps[idx].split("=")
                word2, tag2 = maps[idx+1].split("=")
            else:
                word1, tag1 = maps[idx].split("=")
            tagging.append(tag1)

            if(idx==len(maps)-1 and (tag1 == 'I-PR' or tag1 == 'I-TE' or tag1 == 'I-TR')):
                NLPmodule.append_element(idx, tag1,PR2, TE2, TR2)
            if(tag1 == 'B-PR' or tag1 == 'B-TE' or tag1 == 'B-TR'):
                NLPmodule.append_element(idx, tag1,PR2, TE2, TR2)
                if(tag2 != 'I-PR' and tag2 != 'I-TE' and tag2 != 'I-TR' and (idx < len(maps)-1)):
                    NLPmodule.append_element(idx, tag1,PR2, TE2, TR2)
            elif((tag1 == 'I-PR' or tag1 == 'I-TE' or tag1 == 'I-TR') and (tag2 != 'I-PR' and tag2 != 'I-TE' and tag2 != 'I-TR') and (idx < len(maps)-1)):
                NLPmodule.ppend_element(idx, tag1,PR2, TE2, TR2)

        for i in range(0,len(PR2),2):
            PR.append(tuple(PR2[i:i+2]))
        for i in range(0,len(TE2),2):
            TE.append(tuple(TE2[i:i+2]))
        for i in range(0,len(TR2),2):
            TR.append(tuple(TR2[i:i+2]))

        print(tagging)
        print("PR =",PR)
        print("TE =",TE)
        print("TR =",TR)


# In[6]:


instance = NLPmodule()
input_list = ['I', 'have', 'a', 'headache']
instance.input_val = input_list
print(input_list)

instance.GetNerToken(input_list)

