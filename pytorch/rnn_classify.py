'''
This is my modification of the pytorch tutorial of the following tutorial
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
'''

#%%
# Import dependencies
from __future__ import unicode_literals, print_function, division
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

#%%
# Open the zip file
import zipfile
with zipfile.ZipFile('pytorch/data.zip', 'r') as zip_ref:
  zip_ref.extractall('pytorch')

#%%
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#%%
def unicode_to_ascii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters  
  )

#%%
def read_file(filename):
  lines = open(filename, encoding='utf-8').read().strip().split('\n')
  return [unicode_to_ascii(line) for line in lines]

#%%
X = []
y = []

for filename in glob.glob('pytorch/data/names/*.txt'):
  category = os.path.splitext(os.path.basename(filename))[0]
  X_category = read_file(filename)
  X += X_category
  y += [category] * len(X_category)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%%
def letter_to_index(c):
  return all_letters.find(c)

def letter_to_tensor(c):
  tensor = torch.zeros(1, n_letters)
  tensor[0][letter_to_index(c)] = 1
  return tensor

def name_to_tensor(s):
  tensor = torch.zeros(len(s), 1, n_letters)
  for idx,letter in enumerate(s):
    tensor[li][0][letter_to_index(letter)] = 1
  return tensor

#%%
