# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:58:31 2022

@author: 16028
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
##################################################################################
# credit to PyTorch tutorial--relied on heavily to perform the character classification
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial
##################################################################################

#################################################################################
def findFiles(path):
    return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines


category_lines['win'] = w1
category_lines['lose'] = l1
all_categories = ['win', 'lose']
n_categories = len(all_categories)

#################################################################################

print(category_lines['win'][:5])

#################################################################################


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

###################################################################################
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

##################################################################################
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)
####################################################################################
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)


#################################################################################
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))


#################################################################################
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    # print(line)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
  
    
#################################################################################
criterion = nn.NLLLoss()

#################################################################################
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()





#################################################################################
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
     
################################################################################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)




#################################################################################
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

####################################################################################


def predict(input_line, n_predictions=2): #changed n_predictions from 3 to 2
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
            
            
            
            
            
import numpy as np         
            
def getPrediction(input_line, n_predictions=2): #changed n_predictions from 3 to 2
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        print(output)
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        print(topi)
        predictions = []
        return (np.exp(output[0][0].item()))#, np.exp(output[0][1].item()))
def collectAllPredictions(test_set, n_predictions=2):
    tuples_list = []
    for name in test_set:
        tuples_list.append((name, getPrediction(name)))
    return tuples_list


test_set = X_test['losing_name'][5:15]
collectAllPredictions(test_set)


al = predict('Dovesky')
predict('Jackson')
predict('Satoshi')

predict('Krcatovic')



  


state_to_region = {
    "Alabama": "south",
    "Alaska": "distant",
    "Arizona": "southwest",
    "Arkansas": "south",
    "California": "west_coast",
    "Colorado": "southwest",
    "Connecticut": "northeast",
    "Delaware": "northeast",
    "Florida": "south",
    "Georgia": "south",
    "Hawaii": "distant",
    "Idaho": "middle",
    "Illinois": "rust_belt",
    "Indiana": "rust_belt",
    "Iowa": "middle",
    "Kansas": "middle",
    "Kentucky": "middle",
    "Louisiana": "south",
    "Maine": "northeast",
    "Maryland": "northeast",
    "Massachusetts": "northeast",
    "Michigan": "rust_belt",
    "Minnesota": "middle",
    "Mississippi": "south",
    "Missouri": "south",
    "Montana": "south",
    "Nebraska": "south",
    "Nevada": "west",
    "New Hampshire": "northeast",
    "New Jersey": "northeast",
    "New Mexico": "southwest",
    "New York": "northeast",
    "North Carolina": "border",
    "North Dakota": "middle_west",
    "Ohio": "rust_belt",
    "Oklahoma": "west",
    "Oregon": "west_coast",
    "Pennsylvania": "rust_belt",
    "Rhode Island": "northeast",
    "South Carolina": "south",
    "South Dakota": "middle_west",
    "Tennessee": "south",
    "Texas": "south",
    "Utah": "west",
    "Vermont": "northeast",
    "Virginia": "middle",
    "Washington": "west_coast",
    "West Virginia": "middle",
    "Wisconsin": "rust_belt",
    "Wyoming": "middle_west",
    "District of Columbia": "north_east",
    "American Samoa": "distant",
    "Guam": "distant",
    "Northern Mariana Islands": "distant",
    "Puerto Rico": "distant",
    "United States Minor Outlying Islands": "distant",
    "U.S. Virgin Islands": "distant",
}
