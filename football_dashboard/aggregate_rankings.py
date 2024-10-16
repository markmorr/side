# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 23:21:29 2022

@author: 16028
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import string
import os
import re 


df_dict = dict()
###########################################
# date_old_to_use = "11_28_2023"
# date_to_use = "10_6_2024"
date_to_use = "2024_10_16"
ratcliffe_date_to_use = date_to_use
boone_date_to_use = date_to_use
############################################


ecr_weight = .17 #23
wolf_weight = .21 # 26
fitz_weight = .08 #decay this now--check week-to-week change. likely not updating
boone_weight = .26 #.26
e_q_weight = .17 # this is the weaker_ecr weight



base_path = r'C:\Users\16028\OneDrive\Documents\football_analytics'

dfd_rb = pd.read_csv(base_path + '\\boone_rb_' + boone_date_to_use + '.csv')
dfd_wr = pd.read_csv(base_path + '\\boone_wr_' + boone_date_to_use + '.csv')
dfd_te = pd.read_csv(base_path + '\\boone_te_' + boone_date_to_use + '.csv')
dfd_qb = pd.read_csv(base_path + '\\boone_qb_' + boone_date_to_use + '.csv')



boone_list = [dfd_rb, dfd_wr, dfd_te,dfd_qb]


#gonna have to delete stuff with the team names in parentheses
# as does the wolf
# fitz gets a little extra
# jahnke gets a downgrade

dfd = pd.concat(boone_list)
dfd['rank_boone'] = dfd['PPR'].rank(method='average', ascending=False)
dfd.sort_values(by=['rank_boone'], inplace=True, ascending=True)
dfd.rename(columns={'PLAYER':'name', 'PPR': 'ppr'},inplace=True)
dfd = dfd[['name','rank_boone','ppr']]
dfd['name'] = dfd['name'].str.strip()
dfd['name'] = dfd['name'].str.lower()


 
ecr = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\football_analytics\FantasyPros_2024_Ros_ALL_Rankings_' +
                  date_to_use +'.csv')
ecr
ecr.rename(columns={'PLAYER NAME': 'name', 'RK': 'rank'}, inplace=True)
df_dict['ecr'] = ecr.copy()



weaker_ecr = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\football_analytics\weaker_ecr_' + date_to_use +'.csv')
weaker_ecr.rename(columns={'PLAYER NAME': 'name', 'RK': 'rank'}, inplace=True)
weaker_ecr = weaker_ecr[['name', 'rank', 'SOS PLAYOFFS']]
weaker_ecr['SOS PLAYOFFS'] = weaker_ecr['SOS PLAYOFFS'].astype(str).str[0]
df_dict['weaker_ecr'] = weaker_ecr.copy()
weaker_ecr


df_dict['fitz'] = pd.read_csv(base_path + '\\fitzmaurice_' + date_to_use + '.csv')
df_dict['fitz'] = df_dict['fitz'][['rank', 'name','pos', 'team', 'bye_week',
                               'ecr', 'vs_ecr']]

df_dict['wolf'] = pd.read_csv(base_path + '\\the_wolf_' + date_to_use + '.csv')
df_dict['wolf'] = df_dict['wolf'][['rank', 'name','pos', 'team', 'bye_week',
                               'ecr', 'vs_ecr']]

df_dict['fitz']['pos'] = df_dict['fitz']['pos'].str[:2]
df_dict['wolf']['pos'] = df_dict['wolf']['pos'].str[:2]

df_dict['boone'] = dfd.copy()
for k,v in df_dict.items():
    print(k.upper())
    print(v.columns)
    print('\n') 



dfc = df_dict['fitz'].copy()
dfe = df_dict['ecr'].copy()


qb_list = ['geno smith', 'derek carr', 'aaron rodgers', 'justin fields', 'justin herbert',
           'josh allen', 'joe burrow', 'jalen hurts', 'lamar jackson', 'tua tagovailoa', 'tom brady', 'dak prescott',
            'kirk cousins', 'matthew stafford', 'jared goff', 'kyler murray']


# half PPR to PPR adjustment
def update_dataframe(data, position_column_name, column_to_modify):
    # Define the conditions
    condition_1 = (data[position_column_name].isin(['WR', 'TE']))
    # condition_2 = (data['name'].isin(['christian mccaffrey', 'rhamondre stevenson', 'austin ekeler']))

    # Apply the conditions to update the 'rank_jeff' column
    data.loc[condition_1, column_to_modify] *= 0.875 #| condition_2,


# Update the 'rank_jeff' column for the 'wolf' DataFrame
update_dataframe(df_dict['wolf'], "pos", "rank")

dfw = df_dict['wolf'].copy()


# Print the updated DataFrame
print(df_dict['wolf'])

merge_style = 'left'
# df = dfa.merge(df_dict['fitz'], how=merge_style, left_on='name', right_on = 'name', suffixes=('_jeff','_fitz'))
df = df_dict['fitz'].merge(dfa, how=merge_style, left_on='name', right_on = 'name', suffixes=('_fitz','_jeff'))

df_quatro = df.merge(dfd, how=merge_style, left_on='name', right_on = 'name', suffixes=('_triple', '_bo'))
df_cinco = df_quatro.merge(df_dict['wolf'], how=merge_style, left_on='name', right_on='name', suffixes=('_quad', '_wolf'))
df_seis = df_cinco.merge(df_dict['nj'], how=merge_style, left_on='name', right_on='name', suffixes=('_quad', '_nj'))
df_siete = df_seis.merge(df_dict['ecr'], how=merge_style, left_on='name', right_on='name', suffixes=('_cinco', '_ecr'))
df_ocho = df_siete.merge(df_dict['weaker_ecr'], how=merge_style, left_on='name', right_on='name', 
                         suffixes=('_siete', '_weaker_ecr'))

df = df_ocho.copy()
df_ocho['rank_jeff']

####################
# df = df[df['rank_wolf'].notna()]
##################


# df.dropna(inplace=True)
# df_dict['wolf']

# added to to ecr, took away one from fitz
# df.columns
# ecr_weight = .31
# fitz_weight = .21
# boone_weight = .19
# wolf_weight = .13
# e_q_weight = .04
# jahnke_weight = .03
# ratcliffe_weight = .09


# wolf, jeff

# ecr_weight = 1
# wolf_weight = 1
# fitz_weight = 1
# boone_weight = 1
# e_q_weight = 1
# jahnke_weight = 1
# ratcliffe_weight = 1


sum_of_weights = wolf_weight + fitz_weight + boone_weight + jahnke_weight + ratcliffe_weight + ecr_weight + e_q_weight #+ freedman_weight
print(sum_of_weights)



# df['rank_wolf'].fillna(0, inplace=True)
df['rank_wolf'] = df['rank_wolf'].astype(float)


# def fill_na_with_mean(data):
#     # Calculate the mean value of columns that start with "rank_"
#     rank_columns = [col for col in data.columns if col.startswith('rank_')]
#     mean_rank = data[rank_columns].mean().mean()  # Calculate the overall mean

#     # Fill missing values in 'rank_jeff' with the calculated mean
#     data['rank_jeff'].fillna(mean_rank, inplace=True)
# # df['rank_jeff'].fillna(0, inplace=True)
# # fillna_with_mean(df)

df['rank_jeff'][:10]
def fill_na_with_mean(data, column_name):
    # Calculate the mean value of columns that start with "rank_" and don't contain 0 at the same row
    rank_columns = [col for col in data.columns if col.startswith('rank_')]
    
    # Calculate the mean for each row, excluding 0 values
    data['mean_rank'] = data.apply(lambda row: row[rank_columns][row[rank_columns] != 0].mean(), axis=1)
    # Create an indicator column for filled NA values
    data[column_name + '_na_filled'] = data[column_name].isna()
    # Fill missing values in the specified column with the calculated mean
    data[column_name].fillna(data['mean_rank'], inplace=True)
    # Drop the temporary mean_rank column
    data.drop(columns=['mean_rank'], inplace=True)

fill_na_with_mean(df, 'rank_jeff')
fill_na_with_mean(df, 'rank_wolf')
# prin
print(df[df['name']=='kyle pitts']['rank_wolf'])
df.columns

    # np.mean([35,	25,	36,	23,	45,	29.83333333,	15])

# Fill missing values in 'rank_jeff' column for the 'jeff' DataFrame
# fill_na_with_mean(df_dict['jeff'], 'rank_jeff')

# Print the updated DataFrame
# print(df_dict['jeff'])
#conda update anaconda
# conda update spyder=5.4.3


df['avg_rank'] = round(
    (df['rank_wolf']*wolf_weight + 
     df['rank_fitz']*fitz_weight + 
                   df['rank_boone']*boone_weight + 
                   df['rank_nj']*jahnke_weight +
                  df['rank_jeff']*ratcliffe_weight + 
                  df['rank_weaker_ecr']*e_q_weight + 
                  df['rank_ecr']*ecr_weight) / sum_of_weights,1)
    # + df['rank_free']*freedman_weight)



# df['backup_avg_rank'] = (df['rank_jeff'] * .1 + df['rank_nj']*.4 + df['rank']*.4 + df['rank_boone']*.1)
df.sort_values(by='avg_rank', inplace=True)
df.rename(columns={'position':'pos'}, inplace=True)

cleaner_df = df[['name', 'avg_rank', 'rank_ecr', 'rank_wolf', 'rank_fitz', 'rank_boone', 'rank_weaker_ecr', 'rank_jeff',  
                 'rank_nj','pos_quad' ]] #'ppr' 'rank_nj', 'rank_free',
# 'rank_jeff_na_filled', 'rank_wolf_na_filled'

cleaner_df['std'] = round(df[[ 'rank_ecr', 'rank_wolf', 'rank_fitz', 
                        'rank_boone','rank_weaker_ecr', 'rank_jeff', 'rank_nj']].std(axis=1),2) #'rank_free'

# cleaner_df[cleaner_df['name'] == 'kyle pitts']
cleaner_df


df = cleaner_df.copy()
df.reset_index(inplace=True, drop=True)
df['pos_quad']
df['pos_name'] = df['pos_quad'].str[:2] #pos
df['pos_quad']
df['pos_rank'] = df.groupby('pos_name').cumcount() + 1
df['new_pos_rank'] = df['pos_name'] + df['pos_rank'].astype(str)
df.drop(columns=['pos_quad', 'pos_rank'], inplace=True)
df.index = df.index + 1
temp = df[['name', 'avg_rank']]






trade_left_side = ['aj dillon', 'devonta smith', 'eno benjamin']
trade_right_side = ['keenan allen', 'cam akers']
trade_left_side = ['derrick henry', 'mike boone']
trade_right_side = ['dalvin cook', 'miles sanders', 'alexander mattison']

a = df[df['name'].isin(trade_left_side + trade_right_side)]
# cleaner_df[( cleaner_df['name'].isin(trade_left_side) ) or ( cleaner_df['name'].isin(trade_right_side) )]
cleaner_df[ cleaner_df['name'].isin(trade_right_side)]

trade_right_side = ['taysom hill']


df_dict['jeff'][df_dict['jeff']['name'] == 'tee higgins']
myteam_list = ['jalen hurts', 'travis kelce', 
                    'george pickens', 'tyler allgeier', 'travis etienne', 
                    'tank bigsby', 'calvin ridley', 'breece hall', 'javonte williams', 'deandre hopkins',
                    'bijan robinson', 'dandre swift', 'puka nacua', 'kenneth gainwell'
                   ]

# boone_list

top_n = 160
x1 = dfd['rank_boone'][:top_n].copy()
y1 = dfd['ppr'][:top_n].copy()


tv = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\football_analytics\fantasy_football_trade_value_chart.csv')
tv['rank'] = tv.index + 1
tv['ppr'] = tv['PPR']*y1.max()/67
x2 = tv['rank'][:top_n].copy()
y2 = tv['ppr'][:top_n].copy()

x = pd.concat([x1,x2])
y = pd.concat([y1, y2])


for index, x_val in enumerate(x):
    print(str(index) + ": " + str(x_val))
    

for index, i in enumerate(y):
    print(str(index) + ": " + str(i))

# plt.plot(y)
x = x1
y = y1
x.sort_values(ascending=True, inplace=True)
y.sort_values(ascending=False, inplace=True)
curve = np.polyfit(x, y, deg=4)
# curve = np.array([1,2,3,4,5])
# x = np.linspace(0,10,100)

y_preds = [np.polyval(curve, i) for i in x]
plt.plot(x,y,'-o')
plt.grid()
plt.show()
plt.plot(x,y_preds)
plt.grid()
plt.show()

x_actual = df['avg_rank'].copy()
y_actual_preds = [np.polyval(curve, i) for i in x_actual]
plt.plot(x_actual, y_actual_preds)
plt.grid()
plt.show()
y_preds = pd.Series(y_preds).round(decimals=1)

df[df['pos_name']=='QB']['rank_jeff']

df_new_jeffes = df[df['pos_name']=='QB']['avg_rank'].copy()
df_new_jeffes
df_new_jeffes = df_new_jeffes.reset_index()
df_new_jeffes = df_new_jeffes.reset_index()

df_new_jeffes['level_0'] = df_new_jeffes['level_0'] + 1
df_new_jeffes
jeff_qb = jeff_qb[['Average Rank', 'Player', 'Position']]
jeff_qb.rename(columns={'Average Rank': 'jeff_qb_rank'}, inplace=True)
jqb = jeff_qb.merge(df_new_jeffes, left_on='jeff_qb_rank', right_on='level_0').copy()

jqb
jqb.rename(columns={'Player': 'name'}, inplace=True)
jqb.dropna(inplace=True)
jqb.columns
jqb
jqb.columns = jqb.columns.str.lower()
jqb['name'] = jqb['name'].str.strip()
jqb['name'] = jqb['name'].str.lower()
# print(jqb['name'])
jqb['name'] = '|'.join(jqb['name'].tolist()).translate(transtab).split('|')
# jqb['name'] = ' '.join(jqb['name'].str.split()[:2])
# ' '.join(jqb['name'].str.split()[:2])
jqb['name'] = jqb['name'].str.split(' ').str[0] + ' ' + jqb['name'].str.split(' ').str[1]
jqb.rename(columns={'rank':'rank_' + k}, inplace=True)
# jqb['name'] = [p.sub('', x) for x in jqb['name'].tolist()]
print(jqb.head())
jqb_dict = result_dict = dict(zip(jqb['name'], jqb['avg_rank']))


df[25:70]
for name, rank in jqb_dict.items():
    df.loc[df['name'] == name, 'rank_jeff'] = rank
df[25:70][['name', 'rank_jeff']]   

df['avg_rank'] = round(
    (df['rank_wolf']*wolf_weight + 
     df['rank_fitz']*fitz_weight + 
                   df['rank_boone']*boone_weight + 
                   df['rank_nj']*jahnke_weight +
                  df['rank_jeff']*ratcliffe_weight + 
                  df['rank_weaker_ecr']*e_q_weight + 
                  df['rank_ecr']*ecr_weight) / sum_of_weights,1)
    # + df['rank_free']*freedman_weight)



# df['backup_avg_rank'] = (df['rank_jeff'] * .1 + df['rank_nj']*.4 + df['rank']*.4 + df['rank_boone']*.1)
df.sort_values(by='avg_rank', inplace=True)


df_myteam = df[df['name'].isin(myteam_list)].sort_values(by='avg_rank', ascending=True)
len(myteam_list)
df_myteam.shape[0]
assert df_myteam.shape[0] ==  len(myteam_list)


df.to_csv(r"C:\Users\16028\Downloads\rankings_" + date_to_use + ".csv")
df_myteam.to_csv(r"C:\Users\16028\Downloads\rankings_" + date_to_use + "_my_team.csv")


df[df['pos_name']=='RB'].to_csv(r"C:\Users\16028\Downloads\RB_rankings_" + date_to_use + ".csv")
df[df['pos_name']=='WR'].to_csv(r"C:\Users\16028\Downloads\WR_rankings_" + date_to_use + ".csv")
df[df['pos_name']=='TE'].to_csv(r"C:\Users\16028\Downloads\TE_rankings_" + date_to_use + ".csv")
df[df['pos_name']=='QB'].to_csv(r"C:\Users\16028\Downloads\QB_rankings_" + date_to_use + ".csv")



 