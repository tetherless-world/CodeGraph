#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# VERSION SUMMARY

# version 6: small bugfix
# version 5: added example for tokenization and prediction
# version 4: added apex install for mixed precision training 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import torch

# ## Install requirements

# 0 (optional). install apex for mixed presicion support

# In[ ]:



# 1. pip install pytorch-pretrained-bert without internet

# In[ ]:


os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')

# ## Import Bert

# In[ ]:


from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

# In[ ]:


BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'

# 2. create BERT model and put on GPU

# In[ ]:


bert = BertModel.from_pretrained(BERT_FP).cuda()
bert.eval()

# ## Setup tokenizer

# In[ ]:


tokenizer = BertTokenizer(vocab_file='../input/torch-bert-weights/bert-base-uncased-vocab.txt')

# ## Make prediction

# In[ ]:


# lets tokenize some text (I intentionally mispelled 'plastic' to check berts subword information handling)
text = 'hi my name is Dieter and I like wearing my yellow pglastic hat while coding.'
tokens = tokenizer.tokenize(text)
tokens

# In[ ]:


# added start and end token and convert to ids
tokens = ["[CLS]"] + tokens + ["[SEP]"]
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids

# In[ ]:


# put input on gpu and make prediction
bert_output = bert(torch.tensor([input_ids]).cuda())
bert_output

# ## (Optional) Convert model to fp16

# In[ ]:


import apex
bert.half()

# In[ ]:



