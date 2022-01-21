# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# X_selected_df = pd.DataFrame(X_selected, columns=[X_train.columns[i] for i in range(len(X_train.columns)) if feature_selector.get_support()[i]])
# X_imputed_df = pd.DataFrame(X_imputed, columns = X_train.columns)


# try with grus
# try encoding name--language of origin
# name--gender
# and use those as hard-coded features while also adding the rnn strictly trained to
# predict outcome
# try and combine the signal from those different rnn outputs by inputting them as features
# into a broader modell--use catboost maybe? could run on gpu for instance
# see https://catboost.ai/ video is good
# moved state to region to rnn_elections file
#decision trees are stupid and don't even compare variables
#there are data issues? check what's the one that doesn't fit
#maybe they all fit?
#gonna have to trim names down? give a score to how many times they won, etc.?
# stratify by time and do bag of races--test the performance in both train/test settings
# see if there are any time periods when names perform well, for instance using bag of races



def plotCM(cm_, title_):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm_)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Positive', 'Predicted Negative'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Positive', 'Actual Negative'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_[i, j], ha='center', va='center', color='red')
            plt.title(title_)
    plt.show()
    
    

def runModels(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
    clf_list = [DecisionTreeClassifier(random_state=0,),
    RandomForestClassifier(random_state=0, max_depth=15),
    LogisticRegression(random_state=0, max_iter=10000)]
    for clf in clf_list:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        training_pred = clf.predict(X_train)
        cm = confusion_matrix(y_test, clf.predict(X_test))
        title = clf.__class__.__name__
        print(title)
        # plotCM(cm, title + ' on Testing Data')
        print("Accuracy score: ",accuracy_score(y_test,y_pred))
        print("Balanced accuracy score: ",balanced_accuracy_score(y_test,y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_train,training_pred))
        print(accuracy_score(y_test,y_pred))
        # tree.plot_tree(clf()
    return


df = pd.read_csv(r'C:\Users\16028\Downloads\house_76_thru_2020\1976-2020-house.csv', encoding='unicode_escape')

with open(r'C:\Users\16028\Downloads\house_76_thru_2020\1976-2020-house.csv') as f:
    print(f)



df_filtered = df.groupby(['year', 'state', 'district'])

i = 0
for name, group in df_filtered:
    if i < 10:
        print(name)
        print(group)
        i += 1



df = df[['year', 'state', 'district', 'office', 'stage', 'runoff', 'special', 'candidate', 
         'party', 'candidatevotes', 'totalvotes', 'unofficial', 'version', 'fusion_ticket']]
df['state'] = df['state'].str.lower()
# df['winner'] = df_filtered['candidatevotes'].transform('max')['candidate']
idx = df_filtered['candidatevotes'].transform('max') == df['candidatevotes']
df['winners'] = df[idx]['candidate']

df['winners_fixed'] = df.groupby(['year', 'state', 'district'])['winners'].apply(lambda x: x.ffill().bfill())


df['vote_pct'] = df['candidatevotes']/df['totalvotes']

df = df[df.stage == 'gen']
df = df[df['vote_pct'] > .05]

duo = df.groupby(['year', 'state', 'district']).filter(lambda x:len(x)==2)
duo.reset_index(inplace=True)


df = duo.copy()
df_dems = df[df['party'] == 'DEMOCRAT']
df_non_dems = df[df['party'] != 'DEMOCRAT']
df_m1_alt = df_dems.merge(df_non_dems, left_on=['year', 'state', 'district'], right_on = ['year', 'state', 'district'])

df2 = df_m1_alt[['year', 'state', 'district', 'candidate_x', 'candidate_y', 'party_x',
                 'party_y', 
                 'candidatevotes_x', 'candidatevotes_y', 'totalvotes_x','vote_pct_x','vote_pct_y',
                 'winners_fixed_x']]


def binarize_result(row):
    x_won_flag = False
    if row['candidatevotes_x'] > row['candidatevotes_y']:
        x_won_flag = True
    else:
        x_won_flag = False
    
    if (row['party_x'] == 'DEMOCRAT') & (x_won_flag):
        return 1
    elif (row['party_x'] == 'REPUBLICAN') & (x_won_flag):
        return 0
    elif (row['party_y'] == 'DEMOCRAT') & ( not x_won_flag):
        return 1
    elif (row['party_y'] == 'REPUBLICAN') & (not x_won_flag):
        return 0
    else:
        return 2
# df['y'] = df[['party_x', 'party_y', 'candidatevotes_x', 'candidatevotes_y']].apply(binarize_result)

df2['y']= df2.apply(binarize_result, axis=1)
# df.columns
df2 = df2[df2.y != 2] #for now we'll ignore error cases, again filtering
df = df2.copy()


def binarize_party(x):
    if x =='DEMOCRAT':
        return 1
    elif x == 'REPUBLICAN':
        return 0
    else:
        return 0 #DEFAULTING TO NON-STANDARD MEANS REPUBLICAN
print('here')
df['party_x_bin']= df['party_x'].apply(binarize_party)
df['party_y_bin']= df['party_y'].apply(binarize_party)

df['vote_diff'] = df['candidatevotes_x'] - df['candidatevotes_y']
X = df[['candidatevotes_x', 'candidatevotes_y', 'vote_diff','party_x_bin', 'party_y_bin']]
y = df['y']
X = df[['candidate_x', 'candidate_y', '']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=0)

runModels(X,y)

df.y.value_counts()


df[(df.party_x_bin == 1) & (df.y == 1)].shape
df.shape

def getWinningName(row):
    if row['y'] == 1:
        return row['candidate_x']
    else:
        return row['candidate_y']
    
def getLosingName(row):
    if row['y'] == 1:
        return row['candidate_y']
    else:
        return row['candidate_x']
    

X = df[['candidate_x', 'candidate_y',]]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=0)

vadded = pd.concat([X_train, y_train], axis=1)

def categorizeNames(df__, y_values):
    df__ = pd.concat([df__, y_values], axis=1)
    df__['winning_name'] = df__.apply(getWinningName, axis=1)
    df__['losing_name'] = df__.apply(getLosingName, axis=1)

    df__ = df__[(df__['winning_name'].notna()) & df__['losing_name'].notna()]
    wn = df__['winning_name'].tolist()
    ln = df__['losing_name'].tolist()
    
    
    wn = df__['winning_name'].str.title()
    ln = df__['losing_name'].str.title()
    df__.drop(columns=['y'], axis=1, inplace=True)
    return wn, ln
w1, l1 = categorizeNames(X_train, y_train)



# state_to_region =  {k.lower(): v for k, v in state_to_region.items()}
# df['region'] = df['state'].map(state_to_region)

df['year_delta'] = df['year'] - 1976
one_hot_data = pd.get_dummies(df[['state']],drop_first=False) #probably negligible
numeric_data = df[['year_delta', 'party_x_bin', 'party_y_bin']] #DOES THIS MATTER?
# do i need to double every column because its for the x candidate and the y candidate? 
# should i scramble the order so its not always x is democrat right is republican??
# hmmm??
numeric_data.shape
X = pd.concat([numeric_data, one_hot_data], axis=1)
runModels(X,y)

df.y.value_counts()[1]/(df.y.value_counts()[0]+df.y.value_counts()[1])


one_hot_data = pd.get_dummies(df[['region', ]],drop_first=False) #probably negligible
numeric_data = df[['year_delta']]
numeric_data.shape
X = pd.concat([numeric_data, one_hot_data], axis=1)
runModels(X,y)


one_hot_data = pd.get_dummies(df[['state', ]],drop_first=False) #probably negligible
numeric_data = df[['year_delta']]
numeric_data.shape
X = pd.concat([numeric_data, one_hot_data], axis=1)
runModels(X,y)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
#state only is doing best? logistic regression is learning; deicsion tree is not
#set up a logging system for testing?

#region hurt

# abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))


# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:58:31 2022

@author: 16028
"""


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

n_categories = len(all_categories)

#################################################################################

print(category_lines['Italian'][:5])

#################################################################################
import torch

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


def predict(input_line, n_predictions=3):
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

predict('Dovesky')
predict('Jackson')
predict('Satoshi')


