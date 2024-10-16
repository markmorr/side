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


xy = "This is a sentence. (once a day) [twice a day]"
re.sub("[\(\[].*?[\)\]]", "", xy)

print(os.getcwd())

punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'   # `|` is not present here
transtab = str.maketrans(dict.fromkeys(punct, ''))





df_dict = dict()
###########################################
# was 12_13
# date_old_to_use = "11_28_2023"
date_to_use = "10_6_2024"
ratcliffe_date_to_use = date_to_use
boone_date_to_use = date_to_use
############################################

# google_drive_id = 


ecr_weight = .17 #23
wolf_weight = .21 # 26
fitz_weight = .08 #decay this now--check week-to-week change. likely not updating
boone_weight = .26 #.26
e_q_weight = .17 # this is the weaker ecr
jahnke_weight = .21
ratcliffe_weight = .20 # was .14, make higher now that year almost over

# URL = "https://realpython.github.io/fake-jobs/"
# page = requests.get(URL)
# print(page.text)

#  14*.25 + 10.6*.22 + 13*.06 + 8.5*.25 + 6*.19 + 3*.16 + 10*.13
     #  25 + 22 + 6+ 25 + 19 + 13 + 16
# 11.657/1.26
# df_dict['boone'][:95]
# 84*.75
# url = "https://www.pff.com/news/fantasy-football-rest-of-season-rankings-following-nfl-week-5"
# page = requests.get(url)
# print(page.text)
# internal = page.text
# soup = BeautifulSoup(page.content, "html.parser")
# results = soup.find(id="ResultsContainer")
# print(results.prettify())
# page.text

# job_elements = results.find_all("div", class_="card-content")

# for job_element in job_elements:
#     print(job_element, end="\n"*2)
    
    
# for job_element in job_elements:
#     title_element = job_element.find("h2", class_="title")
#     company_element = job_element.find("h3", class_="company")
#     location_element = job_element.find("p", class_="location")
#     print(title_element)
#     print(company_element)
#     print(location_element)
#     print()
    
# # under end google manager, <div id="app">
# #just doing inspect to right click works as well
# # df = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\football_analytics\ratcliffe_flex_9_30.csv')



base_path = r'C:\Users\16028\OneDrive\Documents\football_analytics'


# df_dict["boone"][:22]
# wolf['name'] = wolf['name'].str.replace("[\(\[].*?[\)\]]", "", regex=True)


# dateyy_to_use = "11_15"
# date_old_to_use = "12_6"
dfd_rb = pd.read_csv(base_path + '\\boone_rb_' + boone_date_to_use + '.csv')
dfd_wr = pd.read_csv(base_path + '\\boone_wr_' + boone_date_to_use + '.csv')
dfd_te = pd.read_csv(base_path + '\\boone_te_' + boone_date_to_use + '.csv')
dfd_qb = pd.read_csv(base_path + '\\boone_qb_' + boone_date_to_use + '.csv')


# dfd_rb = pd.read_csv(base_path + '\\boone_rb_' + date_to_use + '.csv')
# dfd_wr = pd.read_csv(base_path + '\\boone_wr_' + date_to_use + '.csv')
# dfd_te = pd.read_csv(base_path + '\\boone_te_' + date_to_use + '.csv')
# dfd_qb = pd.read_csv(base_path + '\\boone_qb_' + date_to_use + '.csv')
dfd_qb.rename(columns={'1QB': 'PPR'}, inplace=True)
boone_list = [dfd_rb[['PLAYER', 'PPR']], dfd_wr[['PLAYER', 'PPR']], dfd_te[['PLAYER', 'PPR']],
              dfd_qb[['PLAYER', 'PPR']]]



# df_dict['wolf'] = pd.read_csv(base_path + '\\the_wolf_' + date_to_use + '.csv')#, encoding='windows-1254')

df_dict['wolf'] = pd.read_csv(base_path + '\\the_wolf_' + date_to_use + '.csv', index_col=[0])#, encoding='windows-1254')


df_dict['wolf'] = pd.read_csv(base_path + '\\the_wolf_' + date_to_use + '.csv', index_col=[0])
df_dict["wolf"].reset_index(inplace=True)
def extractTier(x):
    if len(x) > 3:
        return x
    else:
        return np.nan

# len(df_dict["wolf"]["ranking"].loc[0])
df_dict["wolf"]["tier"] = df_dict["wolf"]["ranking"].apply(extractTier)
df_dict["wolf"]["tier"].fillna(method="ffill", inplace=True)
df_dict["wolf"].dropna(axis=0, thresh=3, inplace=True)
df_dict["wolf"]["ranking"] = df_dict["wolf"]["ranking"].astype(int)


# df_dict["wolf"].columns
# df_dict["wolf"].iloc[1,:]
# df_dict["wolf"].index
# df_dict['wolf'] = pd.read_csv(base_path + '\\the_wolf_' + date_to_use + '.csv', encoding='windows-1254')
an = df_dict['wolf']
df_dict['wolf'] = df_dict['wolf'][df_dict['wolf']['name'].notna()]# = df[df['EPS'].notna()]
df_dict['wolf'].rename(columns={'ranking':'rank'}, inplace=True)



# df_dict["wolf"]
##########################################################
wtemp = df_dict["wolf"].copy()

wtemp["tier"] = wtemp["tier"].astype(str)  # Convert "tier" to string type if needed
wtemp["rank"] = wtemp["rank"].astype(float)  # Convert "rank" to float type if needed
# wtemp[["name", "rank", "new_rating"]][-50:]

wolf_tiers = df_dict["wolf"].groupby("tier")["rank"].mean()
wolf_tiers_dict = wolf_tiers.to_dict()
wtemp["tier_avg"] = wtemp["tier"].map(wolf_tiers_dict)
wtemp["rank"] = wtemp["rank"]*.8 + wtemp["tier_avg"]*.2
df_dict["wolf"] = wtemp
##########################################################
df_dict['wolf']['name'] = df_dict['wolf']['name'].str.replace("[\(\[].*?[\)\]]", "",regex=True)
df_dict['wolf'] = df_dict['wolf'][['rank', 'name', 'pos', 'bye', 'ecr', 'vs_ecr']]


# data = pd.read_csv(base_path + '\\the_wolf_' + date_to_use + '.csv', encoding='windows-1254')
# data = data[['ranking', 'name', 'pos', 'bye', 'ecr', 'vs_ecr']]
# data.dropna(inplace=True)
# data.rename(columns={'ranking':'rank'}, inplace=True)

# df_dict['wolf'].dropna(inplace=True)

#gonna have to delete stuff with the team names in parentheses
# as does the wolf
# fitz gets a little extra
# jahnke gets a downgrade

# for data in boone_list:
#     print(data.head())
#     data = data[['PLAYER', 'PPR']].copy()
#     data.columns = data.columns.str.lower()


dfd = pd.concat(boone_list)
dfd['rank_boone'] = dfd['PPR'].rank(method='average', ascending=False)
dfd.sort_values(by=['rank_boone'], inplace=True, ascending=True)
dfd.rename(columns={'PLAYER':'name', 'PPR': 'ppr'},inplace=True)
dfd = dfd[['name','rank_boone','ppr']]
dfd['name'] = dfd['name'].str.strip()
dfd['name'] = dfd['name'].str.lower()




# myguy = jeff.concat(jeff_qb)

 # was 2022
 
ecr = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\football_analytics\FantasyPros_2023_Ros_ALL_Rankings_' +
                  date_to_use +'.csv')
ecr.rename(columns={'PLAYER NAME': 'name', 'RK': 'rank'}, inplace=True)
df_dict['ecr'] = ecr.copy()
ecr


weaker_ecr = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\football_analytics\weaker_ecr_' + date_to_use +'.csv')
weaker_ecr.rename(columns={'PLAYER NAME': 'name', 'RK': 'rank'}, inplace=True)
weaker_ecr = weaker_ecr[['name', 'rank', 'SOS PLAYOFFS']]
weaker_ecr['SOS PLAYOFFS'] = weaker_ecr['SOS PLAYOFFS'].astype(str).str[0]
df_dict['weaker_ecr'] = weaker_ecr.copy()
weaker_ecr

###
jeff_qb = pd.read_csv(base_path + '\\ratcliffe_qb_' + ratcliffe_date_to_use + '.csv')
df_dict['jeff'] = pd.read_csv(base_path + '\\ratcliffe_flex_' + ratcliffe_date_to_use + '.csv')
jeff = df_dict['jeff'].copy()
###
df_dict['nj'] = pd.read_csv(base_path + '\\jahnke_' + date_to_use + '.csv')
df_dict['fitz'] = pd.read_csv(base_path + '\\fitzmaurice_' + date_to_use + '.csv')
# df_dict['free'] = pd.read_csv(base_path + '\\freedman_' + date_to_use + '.csv')

df_list =[ df_dict['jeff'], df_dict['nj'], df_dict['fitz'], dfd]

df_dict['fitz'].rename(columns={'player':'name'}, inplace=True)
df_dict['fitz'] = df_dict['fitz'][['rank', 'name','pos', 'team', 'bye_week',
                               'ecr', 'vs_ecr']]
# df_dict['free'] = df_dict['free'][['rank', 'name','pos', 'team', 'bye_week',
#                                'ecr', 'vs_ecr']]

# df_dict['fitz'] = df_dict['fitz'][['rank', 'name','pos', 'team', 'bye_week',
#                                'ecr']]

df_dict['boone'] = dfd.copy()


# df_dict['jeff'] = df_dict['jeff'].rename(columns={'jeff':'rank'})
# df_dict['jeff'] = df_dict['jeff'].rename(columns={'player':'name'})
# df_dict['jeff'].drop(columns=['average rank'], inplace=True)
# df_dict['jeff']

df_dict['jeff']

# df_dict['jeff']
# df_dict['boone']
# df_dict['wolf']
df_dict['jeff'].rename(columns={'Player':'name'}, inplace=True)
df_dict['jeff'].rename(columns={'Average Rank':'rank'}, inplace=True)
# df_dict['jeff']
# ra
df_dict['jeff'] = df_dict['jeff'][['rank', 'name', 'Position', 'Team', 'Jeff']]
for k,v in df_dict.items():
    print(k)
    v.dropna(inplace=True)
    v.columns = v.columns.str.lower()
    v['name'] = v['name'].str.strip()
    v['name'] = v['name'].str.lower()
    # print(v['name'])
    v['name'] = '|'.join(v['name'].tolist()).translate(transtab).split('|')
    # v['name'] = ' '.join(v['name'].str.split()[:2])
    # ' '.join(v['name'].str.split()[:2])
    v['name'] = v['name'].str.split(' ').str[0] + ' ' + v['name'].str.split(' ').str[1]
    v.rename(columns={'rank':'rank_' + k}, inplace=True)
    # v['name'] = [p.sub('', x) for x in v['name'].tolist()]
    print(v.head())
    
   

for k,v in df_dict.items():
    print(v.columns)    
    
# df_dict['jeff'].columns
# ' '.join('travis etienne jr'.split()[:2])



# ' '.join(v['name'].str.split()[:2])


# 'travis etienne jr'.split()[:2]
df_dict["wolf"]





dfb = df_dict['nj'].copy()
dfc = df_dict['fitz'].copy()
dfd = df_dict['boone'].copy()
dfe = df_dict['ecr'].copy()


# dfz = df_dict['free'].copy()




qb_list = ['geno smith', 'derek carr', 'aaron rodgers', 'justin fields', 'justin herbert',
           'josh allen', 'joe burrow', 'jalen hurts', 'lamar jackson', 'tua tagovailoa', 'tom brady', 'dak prescott',
            'kirk cousins', 'matthew stafford', 'jared goff', 'kyler murray']# 'daniel jones'] #'trevor lawrence']

# valuation = 0
# for qb_name in qb_list:
#     valuation = 0
#     for k,v in df_dict.items():
#         # print(k)
#         # print(v[v['name'] == qb_name]['rank_' + k])
#         if k != 'jeff':
#             valuation += int(v[v['name'] == qb_name]['rank_' + k].item())
#     print(qb_name + ': '+ str(valuation/4))
#     # print(valuation/4)

# d[d['name'] == 'geno smith']

# df_dict['wolf'][df_dict['wolf']['name'] == 'geno smith']['rank_wolf'].item()
# dfb[dfb['name'] == 'gabe davis'] 
dfb['name'] = dfb['name'].apply(lambda x: 'gabe davis' if x == 'gabriel davis' else x )
df_dict['nj']['name'] = df_dict['nj']['name'].apply(lambda x: 'gabe davis' if x == 'gabriel davis' else x )

# dfe = df_dict['nj2'].copy()
# dff = df_dict['fitz2'].copy()
# swift, amon ra st. brown
# drake london?
#ezekiel elliot?
# mitchell?

dfd.reset_index(inplace=True)


# df[df['name'] == 'gabe davis']


# dfa
# df = pd.concat([dfa,dfb,dfc], keys='name')
# df_pair = dfa.merge(df_dict['nj'], how='inner', left_on='name', right_on = 'name', suffixes = ('_jeff', '_nj'))

df_dict["wolf"].dtypes
# half PPR to PPR adjustment
def update_dataframe(data, position_column_name, column_to_modify):
    # Define the conditions
    condition_1 = (data[position_column_name].isin(['WR', 'TE']))
    # condition_2 = (data['name'].isin(['christian mccaffrey', 'rhamondre stevenson', 'austin ekeler']))

    # Apply the conditions to update the 'rank_jeff' column
    data.loc[condition_1, column_to_modify] *= 0.875 #| condition_2,

# df_dict['jeff']
# df_dict['wolf']
# Update the 'rank_jeff' column for the 'jeff' DataFrame
update_dataframe(df_dict['jeff'], "position", "rank_jeff")
# Update the 'rank_jeff' column for the 'jeff' DataFrame
update_dataframe(df_dict['wolf'], "pos", "rank_wolf")

df_dict['jeff']

dfa = df_dict['jeff'].copy()
dfw = df_dict['wolf'].copy()

dfa.rename(columns={'Player':'name'}, inplace=True)


# Print the updated DataFrame
print(df_dict['jeff'])
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
# df_nueve = df_ocho.merge(df_dict['free'], how=merge_style, left_on='name', right_on='name', 
#                          suffixes=('_ocho', '_free'))
df = df_ocho.copy()
df_ocho['rank_jeff']
# df_ocho[df_ocho['name']=='kyle pitts']


# df_cinco['kyle pitts']
# df = df_cinco.copy()

# df_dict['fitz']

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

# jeff = df_dict['jeff']




print(6)

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



 