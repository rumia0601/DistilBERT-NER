#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertForTokenClassification
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score

from transformers import AutoTokenizer

#import part


# In[2]:


N = 43823 #columns of dataset
df = pd.read_csv("NER_dataset.csv", encoding="cp949").sample(frac=1)[:N] #load dataset

#change field name
df.rename(columns = {'text':'sentence', 'labels':'tags'}, inplace = True)

#split train, dev, test data (dev data is used for cross validation)
df_train, df_dev, df_test = np.split(df.sample(frac=1, random_state=0), [int(.8 * len(df)), int(.9 * len(df))])


# In[3]:


df.head()
#show head part of dataset


# In[4]:


df.tail()
#show tail part of dataset


# In[5]:


print()

# tansfrom label to list (delimiter is " ")
labels = [str(i).split() for i in df['tags'].values.tolist()]

# count the number of label
unique_labels = set()

for lb in labels:
    [unique_labels.add(i) for i in lb if i not in unique_labels]
 
print(unique_labels)
# {'B-TE', 'I-TR', 'I-PR', 'I-TE', 'O', 'B-TR', 'B-PR'}

# mappingg unique_lables to their id
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
print(labels_to_ids)
# {'B-PR': 0, 'B-TE': 1, 'B-TR': 2, 'I-PR': 3, 'I-TE': 4, 'I-TR': 5, 'O': 6}


# In[6]:


#main class for doing NER with Distilbert

class DistilbertNER(nn.Module):
  
    def __init__(self, tokens_dim): #constructor 
        
        super(DistilbertNER,self).__init__()
        #constructor of parent
    
        if type(tokens_dim) != int:
            raise TypeError('tokens_dim should be an integer')
        if tokens_dim <= 0:
            raise ValueError('Classification layer dimension should be at least 1')
        #exception handling part
            
        self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = tokens_dim) #for using distilBERT model    

    def forward(self, input_ids, attention_mask, labels = None): #define the way for getting ouput by given input
        if labels == None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out #return without label

        else: #labels != None
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
            return out #return with label


# In[7]:


#generalized function for get information from dataset

class NerDataset(torch.utils.data.Dataset):
  
  def __init__(self, df):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')
    
    if "tags" not in df.columns or "sentence" not in df.columns:
      raise ValueError("Dataframe should contain 'tags' and 'sentence' columns")
    
    tags_list = [str(i).split() for i in df["tags"].values.tolist()]
    texts = df["sentence"].values.tolist()
    
    #for change float(nan) -> string("nan")
    i = 0
    for string in texts:
        i += 1
        if(isinstance(string, float)):
            texts[i - 1] = "nan"

    self.texts = [tokenizer(text, padding = "max_length", truncation = True, return_tensors = "pt") for text in texts]
    self.labels = [match_tokens_labels(text, tags) for text,tags in zip(self.texts, tags_list)]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    batch_labels = self.labels[idx]

    return batch_text, torch.LongTensor(batch_labels)


# In[8]:


class MetricsTracking():
  def __init__(self):

    self.total_acc = 0
    self.total_f1 = 0
    self.total_precision = 0
    self.total_recall = 0

  def update(self, predictions, labels , ignore_token = -100):  
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    predictions = predictions[labels != ignore_token]
    labels = labels[labels != ignore_token]

    predictions = predictions.to("cpu")
    labels = labels.to("cpu")

    acc = accuracy_score(labels,predictions)
    f1 = f1_score(labels, predictions, average = "macro")
    precision = precision_score(labels, predictions, average = "macro")
    recall = recall_score(labels, predictions, average = "macro")

    self.total_acc  += acc
    self.total_f1 += f1
    self.total_precision += precision
    self.total_recall  += recall

  def return_avg_metrics(self,data_loader_size):
    n = data_loader_size
    metrics = {
        "acc": round(self.total_acc / n ,3), 
        "f1": round(self.total_f1 / n, 3), 
        "precision" : round(self.total_precision / n, 3), 
        "recall": round(self.total_recall / n, 3)
          }
    return metrics


# In[9]:


#create label

def tags_2_labels(tags : str, tag2idx : dict):
  return [tag2idx[tag] if tag in tag2idx else unseen_label for tag in str(tags).split()] 


# In[10]:


#map words to tag

def tags_mapping(tags_series : pd.Series):
  if not isinstance(tags_series, pd.Series):
      raise TypeError('Input should be a padas Series')

  unique_tags = set()
  
  for tag_list in df_train["tags"]:
    for tag in str(tag_list).split():
      unique_tags.add(tag)

  tag2idx = {k:v for v,k in enumerate(sorted(unique_tags))}
  idx2tag = {k:v for v,k in tag2idx.items()}

  unseen_label = tag2idx["O"]

  return tag2idx, idx2tag, unseen_label, unique_tags


# In[11]:


#-100 means CLS or PAD

def match_tokens_labels(tokenized_input, tags, ignore_token = -100):
        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(ignore_token)

            else :
                try:
                  reference_tag = tags[word_idx]
                  label_ids.append(tag2idx[reference_tag])
                except:
                  label_ids.append(ignore_token)
            
            previous_word_idx = word_idx

        return label_ids


# In[12]:


#train & evaluation function

def train_loop(model, train_dataset, dev_dataset, optimizer,  batch_size, epochs):
  
  train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  for epoch in range(epochs) : 
    
    train_metrics = MetricsTracking()
    total_loss_train = 0

    model.train() #core function for train

    for train_data, train_label in tqdm(train_dataloader):

      train_label = train_label.to(device)

      mask = train_data['attention_mask'].squeeze(1).to(device)
      input_id = train_data['input_ids'].squeeze(1).to(device)

      optimizer.zero_grad()
      
      output = model(input_id, mask, train_label)
      loss, logits = output.loss, output.logits
      predictions = logits.argmax(dim= -1) 

      train_metrics.update(predictions, train_label)
      total_loss_train += loss.item()

      loss.backward()
      optimizer.step()
    
    model.eval() #core function for evaluation

    dev_metrics = MetricsTracking()
    total_loss_dev = 0
    
    with torch.no_grad():
      for dev_data, dev_label in dev_dataloader:

        dev_label = dev_label.to(device)

        mask = dev_data['attention_mask'].squeeze(1).to(device)
        input_id = dev_data['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask, dev_label)
        loss, logits = output.loss, output.logits

        predictions = logits.argmax(dim= -1)     

        dev_metrics.update(predictions, dev_label)
        total_loss_dev += loss.item()
    
    train_results = train_metrics.return_avg_metrics(len(train_dataloader))
    dev_results = dev_metrics.return_avg_metrics(len(dev_dataloader))

    print(f"TRAIN \nLoss: {total_loss_train / len(train_dataset)} \nMetrics {train_results}\n" ) 
    print(f"VALIDATION \nLoss {total_loss_dev / len(dev_dataset)} \nMetrics{dev_results}\n" )   


# In[13]:


#create mapp between label and tag
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping(df_train["tags"])

#change label to tag (surplus label will be changed to "O" tag)
for df in [df_train, df_dev, df_test]:
  df["labels"] = df["tags"].apply(lambda tags : tags_2_labels(tags, tag2idx))


# In[14]:


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") #needed to useing distilBERT model


# In[15]:


text = df_train["sentence"].values.tolist()

#for change float(nan) -> string("nan")

i = 0
for string in text:
    i += 1
    if(isinstance(string, float)):
        text[i - 1] = "nan"

text_tokenized = tokenizer(text, padding = "max_length", truncation = True, return_tensors = "pt")

#map tokens to corresponding words
word_ids = text_tokenized.word_ids()


# In[16]:


model = DistilbertNER(len(unique_tags))
learn = False

#determine whether new train & learn is needed or not

try :
    model = torch.load("NER_model", map_location=torch.device('cpu'))
except FileNotFoundError as e : 
    print("MODEL is not exist so new MODEL will be created")
    learn = True

model.eval()


# In[17]:


#set the hyperparameters

train_dataset = NerDataset(df_train)
dev_dataset = NerDataset(df_dev)

lr = 1e-2
optimizer = SGD(model.parameters(), lr=lr, momentum = 0.9)  

#MAIN

parameters = {
    "model": model,
    "train_dataset": train_dataset,
    "dev_dataset" : dev_dataset,
    "optimizer" : optimizer,
    "batch_size" : 16,
    "epochs" : 5
}

if learn == True: #do train & test if NER_model not exist
    train_loop(**parameters)


# In[18]:


if learn == True: #do export the result of train if NER_model not exist
    torch.save(model, "NER_model")

