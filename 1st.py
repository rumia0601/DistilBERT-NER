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


# In[2]:


import pandas as pd
import numpy as np

N = 1_000
df = pd.read_csv("ner.csv", encoding="cp949").sample(frac=1)[:N]

#change columns names
df.rename(columns = {'text':'sentence', 'labels':'tags'}, inplace = True)

#split train, dev , test sets
df_train, df_dev, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])


# In[3]:


import pandas as pd
df2 = pd.read_csv('ner.csv', encoding = "cp949")
df2.head()


# In[4]:


# Split labels based on whitespace and turn them into a list
labels2 = [i.split() for i in df2['labels'].values.tolist()]

# Check how many labels are there in the dataset
unique_labels2 = set()

for lb in labels2:
  [unique_labels2.add(i) for i in lb if i not in unique_labels2]
 
print(unique_labels2)
# {'B-tim', 'B-art', 'I-art', 'O', 'I-gpe', 'I-per', 'I-nat', 'I-geo', 'B-eve', 'B-org', 'B-gpe', 'I-eve', 'B-per', 'I-tim', 'B-nat', 'B-geo', 'I-org'}

# Map each label into its id representation and vice versa
labels_to_ids2 = {k: v for v, k in enumerate(sorted(unique_labels2))}
ids_to_labels2 = {v: k for v, k in enumerate(sorted(unique_labels2))}
print(labels_to_ids2)
# {'B-art': 0, 'B-eve': 1, 'B-geo': 2, 'B-gpe': 3, 'B-nat': 4, 'B-org': 5, 'B-per': 6, 'B-tim': 7, 'I-art': 8, 'I-eve': 9, 'I-geo': 10, 'I-gpe': 11, 'I-nat': 12, 'I-org': 13, 'I-per': 14, 'I-tim': 15, 'O': 16}


# In[5]:


class DistilbertNER(nn.Module):
  """
  Implement NN class based on distilbert pretrained from Hugging face.
  Inputs : 
    tokens_dim : int specifyng the dimension of the classifier
  """
  
  def __init__(self, tokens_dim):
    super(DistilbertNER,self).__init__()
    
    if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

    if tokens_dim <= 0:
          raise ValueError('Classification layer dimension should be at least 1')

    self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = tokens_dim) #set the output of each token classifier = unique_lables


  def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss
    """
  Forwad computation of the network
  Input:
    - inputs_ids : from model tokenizer
    - attention :  mask from model tokenizer
    - labels : if given the model is able to return the loss value
  """

    #inference time no labels
    if labels == None:
      out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
      return out

    out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
    return out


# In[6]:


class NerDataset(torch.utils.data.Dataset):
  """
  Custom dataset implementation to get (text,labels) tuples
  Inputs:
   - df : dataframe with columns [tags, sentence]
  """
  
  def __init__(self, df):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')
    
    if "tags" not in df.columns or "sentence" not in df.columns:
      raise ValueError("Dataframe should contain 'tags' and 'sentence' columns")

     
    
    tags_list = [i.split() for i in df["tags"].values.tolist()]
    texts = df["sentence"].values.tolist()

    self.texts = [tokenizer(text, padding = "max_length", truncation = True, return_tensors = "pt") for text in texts]
    self.labels = [match_tokens_labels(text, tags) for text,tags in zip(self.texts, tags_list)]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    batch_labels = self.labels[idx]

    return batch_text, torch.LongTensor(batch_labels)


# In[7]:


class MetricsTracking():
  """
  In order make the train loop lighter I define this class to track all the metrics that we are going to measure for our model.
  """
  def __init__(self):

    self.total_acc = 0
    self.total_f1 = 0
    self.total_precision = 0
    self.total_recall = 0

  def update(self, predictions, labels , ignore_token = -100):
    '''
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100
    '''  
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


# In[8]:


def tags_2_labels(tags : str, tag2idx : dict):
  '''
  Method that takes a list of tags and a dictionary mapping and returns a list of labels (associated).
  Used to create the "label" column in df from the "tags" column.
  '''
  return [tag2idx[tag] if tag in tag2idx else unseen_label for tag in tags.split()] 


# In[9]:


def tags_mapping(tags_series : pd.Series):
  """
  tag_series = df column with tags for each sentence.
  Returns:
    - dictionary mapping tags to indexes (label)
    - dictionary mappign inedexes to tags
    - The label corresponding to tag 'O'
    - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
  """

  if not isinstance(tags_series, pd.Series):
      raise TypeError('Input should be a padas Series')

  unique_tags = set()
  
  for tag_list in df_train["tags"]:
    for tag in tag_list.split():
      unique_tags.add(tag)


  tag2idx = {k:v for v,k in enumerate(sorted(unique_tags))}
  idx2tag = {k:v for v,k in tag2idx.items()}

  unseen_label = tag2idx["O"]

  return tag2idx, idx2tag, unseen_label, unique_tags


# In[10]:


def match_tokens_labels(tokenized_input, tags, ignore_token = -100):
        '''
        Used in the custom dataset.
        -100 will be tha label used to match additional tokens like [CLS] [PAD] that we dont care about. 
        Inputs : 
          - tokenized_input : tokenizer over the imput text -> {input_ids, attention_mask}
          - tags : is a single label array -> [O O O O O O O O O O O O O O B-tim O]
        
        Returns a list of labels that match the tokenized text -> [-100, 3,5,6,-100,...]
        '''

        #gives an array [ None , 0 , 1 ,2 ,... None]. Each index tells the word of reference of the token
        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(ignore_token)

            #if its equal to the previous word we can add the same label id of the provious or -100 
            else :
                try:
                  reference_tag = tags[word_idx]
                  label_ids.append(tag2idx[reference_tag])
                except:
                  label_ids.append(ignore_token)
              
            
            previous_word_idx = word_idx

        return label_ids


# In[11]:


def freeze_model(model,num_layers = 1):
  """
  Freeze last num_layers of a model to prevent ctastrophic forgetting.
  Doesn't seem to work weel, its better to fine tune the entire netwok
  """
  for id , params in enumerate(model.parameters()):
    if id == len(list(model.parameters())) - num_layers: 
      print("last layer unfreezed")
      params.requires_grad = True
    else:
      params.requires_grad = False
  return model


# In[12]:


def train_loop(model, train_dataset, dev_dataset, optimizer,  batch_size, epochs):
  
  train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  for epoch in range(epochs) : 
    
    train_metrics = MetricsTracking()
    total_loss_train = 0

    model.train() #train mode

    for train_data, train_label in tqdm(train_dataloader):

      train_label = train_label.to(device)
      '''
      squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
      '''
      mask = train_data['attention_mask'].squeeze(1).to(device)
      input_id = train_data['input_ids'].squeeze(1).to(device)

      optimizer.zero_grad()
      
      output = model(input_id, mask, train_label)
      loss, logits = output.loss, output.logits
      predictions = logits.argmax(dim= -1) 

      #compute metrics
      train_metrics.update(predictions, train_label)
      total_loss_train += loss.item()

      #grad step
      loss.backward()
      optimizer.step()
    
    '''
    EVALUATION MODE
    '''            
    model.eval()

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


#create tag-label mapping
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping(df_train["tags"])

#create the label column from tag. Unseen labels will be tagged as "O"
for df in [df_train, df_dev, df_test]:
  df["labels"] = df["tags"].apply(lambda tags : tags_2_labels(tags, tag2idx))


# In[14]:


#original text
text = df_train["sentence"].values.tolist()

#toeknized text
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_tokenized = tokenizer(text , padding = "max_length" , truncation = True, return_tensors = "pt" )

#mapping token to original word
word_ids = text_tokenized.word_ids()


# In[15]:


model = DistilbertNER(len(unique_tags))

try :
    model = torch.load("MODEL")
except FileNotFoundError as e : 
    print("MODEL is not exist so new MODEL will be created")

model.eval()


# In[16]:


#model = DistilbertNER(len(unique_tags))
#Prevent Catastrofic Forgetting
#model = freeze_model(model, num_layers = 2)

#datasets
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
    "epochs" : 8
}

train_loop(**parameters)


# In[17]:


torch.save(model, "MODEL")


# In[18]:


def align_word_ids(texts):
  
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


# In[26]:


def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels2[i] for i in predictions]
    print(sentence)
    print(prediction_label)
    return prediction_label


# In[37]:


text = "I did a blood donation"
text2 = evaluate_one_text(model, text)


# In[ ]:




