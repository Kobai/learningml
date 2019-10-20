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
cat_dict = {}

for idx,filename in enumerate(glob.glob('pytorch/data/names/*.txt')):
  category = os.path.splitext(os.path.basename(filename))[0]
  X_category = read_file(filename)
  cat_dict[str(idx)] = category
  X += X_category
  y += [idx] * len(X_category)

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
    tensor[idx][0][letter_to_index(letter)] = 1
  return tensor

#%%
class Model(nn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(Model, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_size)
  
  def forward(self, x):
    hidden = self.init_hidden()
    out, hidden = self.rnn(x, hidden)
    out = out.continguous().view(-1, self.hidden_dim) 
    out = self.fc(out)
    return out, hidden
  
  def init_hidden(self):
    return torch.zeros(1, self.hidden_dim)

rnn = Model(n_letters, 18, 128, 1)

#%%
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

def step(X,y):
  hidden = rnn.init_hidden()
  optimizer.zero_grad()

  name_tensor = name_to_tensor(X)
  print(X)
  print(name_tensor.size())
  category_tensor = torch.tensor([y], dtype=torch.long)
  
  output, hidden = rnn(name_tensor)
  loss = criterion(output, category_tensor)
  loss.backward()
  optimizer.step()

  return output, loss.item()

#%%
all_loss = []

def get_guess(tensor):
  top_n, top_i = output.topk(1)
  category_i = top_i[0].item()
  return cat_dict[str(category_i)], category_i

def train():
  current_loss = 0
  for i in range(5):
    for idx, (X,y) in enumerate(zip(X_train, y_train)):
      output, loss = step(X,y)
      current_loss += loss
      if idx % 800 == 0:
        guess, guess_i = get_guess(output) 
        print(f'{guess} / {cat_dict[str(y)]} Loss: {current_loss/800}')
        all_loss.append(current_loss / 400)
        current_loss = 0
    print(f'Done epoch({i})')

train()

#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_loss)
plt.show()

#%%
for epoch in range(1):
  for X,y in zip(X_train, y_train):
    step(X,y)


#%%
