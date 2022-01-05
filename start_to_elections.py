# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



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
    
    clf = DecisionTreeClassifier(random_state=0, max_depth=1)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    training_pred = clf.predict(X_train)
    cm = confusion_matrix(y_test, clf.predict(X_test))
    title = clf.__class__.__name__
    print(title)
    plotCM(cm, title + ' on Testing Data')
    print(accuracy_score(y_train,training_pred))
    print(accuracy_score(y_test,y_pred))
    tree.plot_tree(clf)


    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    training_pred = clf.predict(X_train)
    cm = confusion_matrix(y_test, clf.predict(X_test))
    title = clf.__class__.__name__
    print(title)
    plotCM(cm, title + ' on Testing Data')
    print(accuracy_score(y_train,training_pred))
    print(accuracy_score(y_test,y_pred))

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

from sklearn.tree import DecisionTreeClassifier


def binarize_party(x):
    if x =='DEMOCRAT':
        return 1
    elif x == 'REPUBLICAN':
        return 0
    else:
        return 0 #DEFAULTING TO NON-STANDARD MEANS REPUBLICAN
    
df['party_x_bin']= df['party_x'].apply(binarize_party)
df['party_y_bin']= df['party_y'].apply(binarize_party)

df['vote_diff'] = df['candidatevotes_x'] - df['candidatevotes_y']

X = df[['candidatevotes_x', 'candidatevotes_y', 'vote_diff','party_x_bin', 'party_y_bin']]
y = df['y']

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)

clf = DecisionTreeClassifier(random_state=0, max_depth=1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

training_pred = clf.predict(X_train)

print(accuracy_score(y_train,training_pred))
print(accuracy_score(y_test,y_pred))
tree.plot_tree(clf)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

training_pred = clf.predict(X_train)


print(accuracy_score(y_train,training_pred))
print(accuracy_score(y_test,y_pred))
#decision trees are stupid and don't even compare variables
#there are data issues? check what's the one that doesn't fit
#maybe they all fit?


df.y.value_counts()



df[(df.party_x_bin == 1) & (df.y == 1)].shape
df.shape



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


state_to_region =  {k.lower(): v for k, v in state_to_region.items()}

df['region'] = df['state'].map(state_to_region)

df['year_delta'] = df['year'] - 1976
one_hot_data = pd.get_dummies(df[['region', 'state']],drop_first=False) #probably negligible
numeric_data = df[['year_delta']]
numeric_data.shape
X = pd.concat([numeric_data, one_hot_data], axis=1)
runModels(X,y)

#set up a logging system for testing?

#region hurt

# abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))
