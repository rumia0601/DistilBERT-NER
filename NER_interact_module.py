#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import torch
from transformers import AutoTokenizer
from transformers import DistilBertForTokenClassification
import torch.nn as nn
import re
warnings.filterwarnings('ignore')

using_biobert = False #biobert option


# In[2]:


def align_word_ids(texts):
    #Define AutoTokenizer and Set options
    
    if using_biobert == False :
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") #needed to useing distilBERT model
    else :
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", model_max_length = 512) #needed to useing bioBERT model
    
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    #set tokens ID
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []
    #and labeling them
    #append = -100 means "no meaning"
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
                label_ids.append(-100)
        previous_word_idx = word_idx
        
    return label_ids


# In[3]:


def evaluate_one_text(model, sentence):
    
    if using_biobert == False :
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") #needed to useing distilBERT model
    else :
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", model_max_length = 512) #needed to useing bioBERT model
    
    #set tag types based BIO system
    unique_labels2 = set()
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

    #device set(Assume CPU)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #when cuda used
    if use_cuda:
        model = model.cuda()

    #tokenize input data
    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    #take data from DistilbertNER to use model
    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    #do labelling by using model
    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]
    
    #changle make function result
    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels2[i] for i in predictions]
    return prediction_label


# In[4]:


class DistilbertNER(nn.Module):
    def __init__(self, tokens_dim):
        super(DistilbertNER, self).__init__()
        #input type check
        if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')
        #number of input check
        if tokens_dim <= 0:
            raise ValueError(
                'Classification layer dimension should be at least 1')
        #set option to use Model(use from_pretrained)

        if using_biobert == False :
            self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = tokens_dim) #for using distilBERT model
        else :
            self.pretrained = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2", num_labels = tokens_dim) #for using bioBERT model
        
    def forward(self, input_ids, attention_mask, labels=None):  # labels are needed in order to compute the loss

        if labels is None:
            out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
            return out

        out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out


# In[5]:


class NLPmodule:    #class to take and save model
    def __init__(self):
        self.model = torch.load("./NER_model", "cpu")
        #load model based CPU environment
    def append_element(self, idx, tag1, pr2, te2, tr2):
        #make final result list by type
        if tag1 == 'B-PR' or tag1 == 'I-PR':
            pr2.append(idx)
        elif tag1 == 'B-TE' or tag1 == 'I-TE':
            te2.append(idx)
        elif tag1 == 'B-TR' or tag1 == 'I-TR':
            tr2.append(idx)

    def get_ner_token(self, text):
        #use to take user input and run model
        text = ' '.join(text)
        label = evaluate_one_text(self.model, text)
        #make list to string and labeling by using mmodel
        for i in range(len(label)):
            if (i == 0) and (label[i] == 'I-PR'):
                label[i] = 'B-PR'
            elif (i == 0) and (label[i] == 'I-TE'):
                label[i] = 'B-TE'
            elif (i == 0) and (label[i] == 'I-TR'):
                label[i] = 'B-TR'
            elif (i != 0) and (label[i] == 'I-PR') and (
                    label[i - 1] != 'B-PR' and label[i - 1] != 'I-PR'):
                label[i] = 'B-PR'
            elif (i != 0) and (label[i] == 'I-TE') and (
                    label[i - 1] != 'B-TE' and label[i - 1] != 'I-TE'):
                label[i] = 'B-TE'
            elif (i != 0) and (label[i] == 'I-TR') and (
                    label[i - 1] != 'B-TR' and label[i - 1] != 'I-TR'):
                label[i] = 'B-TR'
        # post-processing

        print(text)
        print(label)
        pattern = "[ ]"
        words = re.split(pattern, text)
        for i in range(len(words)):
            words[i] += "="
        maps = list(map(str.__add__, words, label))
        print(maps)
        #make set of text & label

        # change output format
        tagging = []
        pr = []
        te = []
        tr = []
        pr2 = []
        te2 = []
        tr2 = []

        B_tag = ['B-PR','B-TE','B-TR']
        I_tag = ['I-PR','I-TE','I-TR']
        #tag list
        for idx in range(len(maps)):
        #about all elements of text & label set
            if idx < len(maps) - 1:
                word1, tag1 = maps[idx].split("=")
                word2, tag2 = maps[idx + 1].split("=")
                if tag1 in B_tag:
                  self.append_element(idx, tag1, pr2, te2, tr2)
                  if tag2 not in I_tag:
                    self.append_element(idx, tag1, pr2, te2, tr2)
                #to check 1 length case
                elif (tag1 in I_tag) and (tag2 not in I_tag):
                  self.append_element(idx, tag1, pr2, te2, tr2)
            else:
            #at last of list
                word1, tag1 = maps[idx].split("=")
                if tag1 in B_tag:
                  self.append_element(idx, tag1, pr2, te2, tr2)
                  self.append_element(idx, tag1, pr2, te2, tr2)
                elif tag1 in I_tag:
                  self.append_element(idx, tag1, pr2, te2, tr2)
            tagging.append(tag1)
        # post-processing
        for i in range(0, len(pr2), 2):
            pr.append(tuple(pr2[i:i + 2]))
        for i in range(0, len(te2), 2):
            te.append(tuple(te2[i:i + 2]))
        for i in range(0, len(tr2), 2):
            tr.append(tuple(tr2[i:i + 2]))
        
        print(tagging)
        print("PR =", pr)
        print("TE =", te)
        print("TR =", tr)


# instance = NLPmodule()
# #input_list = ["His goal INR should be 1.6-2.0 . If bleeding continues to occur , consider V. Winchester filter ."]
# #input_list = ['Since', 'this', 'patient', 'suffered', 'a', 'heart', 'attack', 'and', 'has', 'been', 'wearing', 'a', 'pacemaker', ',', 'an', 'electromagnetic', 'resonance', 'scan', 'cannot', 'be', 'performed', 'to', 'diagnose', 'lung', 'cancer', '.']
# #input_list = ['Review', 'of', 'the', 'medical', 'record', 'revealed', 'several', 'previous', 'admissions', 'for', 'cellulitis', 'and', 'bacteremia', '.', 'Blood', 'cultures', 'on', 'this', 'admission', 'and', 'were', 'positive', 'for', 'Beta', 'streptococcus', 'Group', 'B', '.', 'The', 'patient', 'had', 'previous', 'episodes', 'of', 'the', 'same', 'infection', ',', 'with', 'lower', 'extremity', 'cellulitis', 'the', 'likely', 'source', '.', 'A', 'left', 'lower', 'extremity', 'ultrasound', 'was', 'negative', 'for', 'deep', 'venous', 'thrombosis', '.', 'The', 'patient', 'was', 'too', 'large', 'to', 'have', 'a', 'CTA', '.', 'He', 'was', 'initially', 'treated', 'with', 'vancomycin', 'and', 'zosyn', 'while', 'awaiting', 'culture', 'results', '.', 'Once', 'the', 'organism', 'was', 'identified', ',', 'he', 'was', 'transitioned', 'to', 'a', 'regimen', 'of', 'intravenous', 'penicillin', 'and', 'oral', 'levofloxacin', ',', 'which', 'had', 'successfully', 'treated', 'the', 'infection', 'during', 'his', 'most', 'recent', 'admission', '1', 'year', 'prior', '.', 'He', 'completed', '7', 'days', 'of', 'levofloxacin', 'and', 'had', 'a', 'PICC', 'line', 'placed', 'to', 'facilitate', 'a', '2', 'week', 'course', 'of', 'IV', 'Penicillin', '.']
# input_list = ['Review', 'of', 'the', 'medical', 'record', 'revealed', 'several', 'previous', 'admissions', 'for', 'cellulitis', 'and', 'bacteremia', '.', 'Blood', 'cultures', 'on', 'this', 'admission', 'and', 'were', 'positive', 'for', 'Beta', 'streptococcus', 'Group', 'B', '.', 'The', 'patient', 'had', 'previous', 'episodes', 'of', 'the', 'same', 'infection', ',', 'with', 'lower', 'extremity', 'cellulitis', 'the', 'likely', 'source', '.', 'A', 'left', 'lower', 'extremity', 'ultrasound', 'was', 'negative', 'for', 'deep', 'venous', 'thrombosis', '.', 'The', 'patient', 'was', 'too', 'large', 'to', 'have', 'a', 'CTA', '.', 'He', 'was', 'initially', 'treated', 'with', 'vancomycin', 'and', 'zosyn', 'while', 'awaiting', 'culture', 'results', '.', 'Once', 'the', 'organism', 'was', 'identified', ',', 'he', 'was', 'transitioned', 'to', 'a', 'regimen', 'of', 'intravenous', 'penicillin', 'and', 'oral', 'levofloxacin', ',', 'which', 'had', 'successfully', 'treated', 'the', 'infection', 'during', 'his', 'most', 'recent', 'admission', '1', 'year', 'prior', '.']
# print(input_list)
# 
# instance.get_ner_token(input_list)

# In[6]:


instance = NLPmodule()

string = "Review of the medical record revealed several previous admissions for cellulitis and bacteremia ."
input_list = string.split(' ')
print(string)
print(input_list)
instance.get_ner_token(input_list)

string = "Blood cultures on this admission and were positive for Beta streptococcus Group B ."
input_list = string.split(' ')
print(string)
print(input_list)
instance.get_ner_token(input_list)

string = "The patient had previous episodes of the same infection , with lower extremity cellulitis the likely source ."
input_list = string.split(' ')
print(string)
print(input_list)
instance.get_ner_token(input_list)

string = "A left lower extremity ultrasound was negative for deep venous thrombosis ."
input_list = string.split(' ')
print(string)
print(input_list)
instance.get_ner_token(input_list)

string = "The patient was too large to have a CTA ."
input_list = string.split(' ')
print(string)
print(input_list)
instance.get_ner_token(input_list)

string = "Review of the medical record revealed several previous admissions for cellulitis and bacteremia . Blood cultures on this admission and were positive for Beta streptococcus Group B . The patient had previous episodes of the same infection , with lower extremity cellulitis the likely source . A left lower extremity ultrasound was negative for deep venous thrombosis . The patient was too large to have a CTA ."
input_list = string.split(' ')
print(string)
print(input_list)
instance.get_ner_token(input_list)


# In[ ]:




