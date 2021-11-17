# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:17:04 2021

@author: 16028
"""

import os
import argparse
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import time
from nltk import word_tokenize
import numpy as np



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




LABELS = ['F', 'T']

def get_wic_subset(data_dir):
	wic = []
	split = data_dir.strip().split('/')[-1]
	with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
		open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
		for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
			word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
			sentence1_word_index, sentence2_word_index = word_indices.split('-')
			label = LABELS.index(label_line.strip())
			wic.append({
				'word': word,
				'sentence1_word_index': int(sentence1_word_index),
				'sentence2_word_index': int(sentence2_word_index),
				'sentence1_words': sentence1.split(' '),
				'sentence2_words': sentence2.split(' '),
				'label': label
			})
	return wic


# =============================================================================
# np_diffs_train = np.array(diffs_train)
# np_diffs_test = np.array(diffs_test)
# 
# 
# 
# def runTest(X_train, y_train, X_test, y_test):    
#     clf = LogisticRegression(random_state=0, max_iter=10000, penalty='none').fit(X_train, y_train)
#     y_preds = clf.predict(X_test)
#     y_training_preds = clf.predict(X_train)
#     print(accuracy_score(y_train, y_training_preds))
#     print(accuracy_score(y_test, y_preds))
# 
# runTest(np_diffs_train, y_train, np_diffs_test, y_test)
# 
# 
# clf = LogisticRegression(random_state=0, max_iter=10000, penalty='none').fit(X_train, y_train)
# y_preds = clf.predict(X_test)
# y_training_preds = clf.predict(X_train)
# print(accuracy_score(y_train, y_training_preds))
# print(accuracy_score(y_test, y_preds))
# 
# =============================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
	description='Train a classifier to recognize words in context (WiC).'
	)
    parser.add_argument(
		'--train-dir',
		dest='train_dir',
		required=True,
		help='The absolute path to the directory containing the WiC train files.'
	)
    parser.add_argument(
		'--eval-dir',
		dest='eval_dir',
		required=True,
		help='The absolute path to the directory containing the WiC eval files.'
	)
	# Write your predictions (F or T, separated by newlines) for each evaluation
	# example to out_file in the same order as you find them in eval_dir.  For example:
	# F
	# F
	# T
	# where each row is the prediction for the corresponding line in eval_dir.
    parser.add_argument(
		'--out-file',
		dest='out_file',
		required=True,
		help='The absolute path to the file where evaluation predictions will be written.')
    args = parser.parse_args()

    ################################# MY CODE ###################################

    out_file_directory = args.out_file
    # word_dict_train = get_wic_subset('wic\train' )

    word_dict_train = get_wic_subset(args.train_dir)
    word_dict_test = get_wic_subset(args.eval_dir)


# When the TAs run the script are the command line arguments going to be formatted 
# such that this is correct?
######


    # or this:
    # word_dict_train = get_wic_subset(args.train_dir )
    # word_dict_test = get_wic_subset(args.eval_dir )
    #######################################################


    # =============================================================================
    # N = len(word_dict_train)
    # sent_list_form_train = []
    # sent_list_form_test = []
    # sent_concat_train = []
    # sent_concat_test = []
    # y_train = []
    # y_test = []
    # 
    # for i in range(N):
    #     sent_list_form_train.append(word_dict_train[i]['sentence1_words'])
    #     sent_list_form_train.append(word_dict_train[i]['sentence2_words'])
    #     sent_concat_train.append(" ".join(word_dict_train[i]['sentence1_words']))
    #     sent_concat_train.append(" ".join(word_dict_train[i]['sentence2_words']))
    # 
    #     y_train.append(word_dict_train[i]['label'])
    #     
    #   
    # y_train = np.array(y_train)
    # for i in range(len(word_dict_test)):
    #     sent_list_form_test.append(word_dict_test[i]['sentence1_words'])
    #     sent_list_form_test.append(word_dict_test[i]['sentence2_words'])
    #     sent_concat_test.append(" ".join(word_dict_test[i]['sentence1_words']))
    #     sent_concat_test.append(" ".join(word_dict_test[i]['sentence2_words']))
    #     y_test.append(word_dict_test[i]['label'])
    # 
    # =============================================================================



    #################################################################################
    #################################################################################
    #################################################################################
    #### TRAIN ##########################################################################

    sent_concat_train = []
    y_train = []

    for i in range(len(word_dict_train)):
        sent_concat_train.append(" ".join(word_dict_train[i]['sentence1_words']))
        sent_concat_train.append(" ".join(word_dict_train[i]['sentence2_words']))

        y_train.append(word_dict_train[i]['label'])
        


    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained("bert-base-cased", output_hidden_states = True)
    encoded_input = tokenizer(sent_concat_train, padding=True, truncation=True,return_tensors='pt') # padding="max_length", truncation=True,

    start = time.time()
    N = len(sent_concat_train)
    i = 0
    inc_num = 5
    ii = []
    tti = []
    am = []
    hidden_states = []

    with torch.no_grad():
        while i < len(sent_concat_train):
            if i%1000 == 0: print(i)
            ii = encoded_input['input_ids'][i:i+inc_num]
            tti = encoded_input['token_type_ids'][i:i+inc_num]
            am = encoded_input['attention_mask'][i:i+inc_num]

            output=model(input_ids=ii, attention_mask=torch.tensor(am), token_type_ids=tti)
            hidden_states.append(output.last_hidden_state)
            i = i + inc_num 
            # 0:5->5:10->10:15 ->45:50->50:55 (i think it's working?')
            

    end = time.time()
    print(round(end-start,1))
    ###################################################################################

    total_list = []
    for tens in hidden_states:
        for i in tens:
            srmw = torch.mean(i, dim=0) 
            total_list.append(srmw)

    sent1_embeds_train = []
    sent2_embeds_train = []

    for i in range(len(total_list)):
        if i%2 == 0: 
            sent1_embeds_train.append(total_list[i])
        else:
            sent2_embeds_train.append(total_list[i])
            
    both_list_train = []
    for i in range(len(sent1_embeds_train)):
        both_list_train.append([sent1_embeds_train[i], sent2_embeds_train[i]])
        

    diffs_train = []
    cosine_sims_train = []
    euclidean_dist_train = []
    l1_dist_train = []
    for i in range(len(sent1_embeds_train)):
        sent1_vec = both_list_train[i][0].reshape(1,-1)
        sent2_vec = both_list_train[i][1].reshape(1,-1)
        cosine_sims_train.append(cosine_similarity(sent1_vec, sent2_vec)[0][0])
        dif = sent1_vec - sent2_vec
        diffs_train.append(torch.squeeze(dif, dim=0))

        euclidean_dist_train.append(np.linalg.norm(sent1_vec - sent2_vec))
        l1_dist_train.append(np.linalg.norm(sent1_vec - sent2_vec, ord=1))  
        
    for i in range(len(diffs_train)):
        diffs_train[i] = np.array(diffs_train[i])



    #################################################################################
    #################################################################################
    #################################################################################




    ###################################################################################
    ###################################################################################
    ###################################################################################
    #### TEST ##########################################################################

    sent_concat_test = []

    y_test = []

    for i in range(len(word_dict_test)):
        sent_concat_test.append(" ".join(word_dict_test[i]['sentence1_words']))
        sent_concat_test.append(" ".join(word_dict_test[i]['sentence2_words']))

        y_test.append(word_dict_test[i]['label'])
        
        


    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained("bert-base-cased", output_hidden_states = True)
    encoded_input = tokenizer(sent_concat_test, padding=True, truncation=True,return_tensors='pt') # padding="max_length", truncation=True,

    start = time.time()
    N = len(sent_concat_test)
    i = 0
    inc_num = 5
    ii = []
    tti = []
    am = []
    hidden_states_test = []
    with torch.no_grad():
        while i < len(sent_concat_test):
            if i%100 == 0: print(i)
            ii = encoded_input['input_ids'][i:i+inc_num]
            tti = encoded_input['token_type_ids'][i:i+inc_num]
            am = encoded_input['attention_mask'][i:i+inc_num]

            output=model(input_ids=ii, attention_mask=torch.tensor(am), token_type_ids=tti)
            hidden_states_test.append(output.last_hidden_state)
            i = i + inc_num 
    end = time.time()
    print(round(end-start,1))


    total_list = []
    for tens in hidden_states_test:
        for i in tens:
            srmw = torch.mean(i, dim=0) 
            total_list.append(srmw)
            
    #################################################################################



    sent1_embeds_test = []
    sent2_embeds_test = []

    for i in range(len(total_list)):
        if i%2 == 0: 
            sent1_embeds_test.append(total_list[i])
        else:
            sent2_embeds_test.append(total_list[i])
            
    both_list_test = []
    for i in range(len(sent1_embeds_test)):
        both_list_test.append([sent1_embeds_test[i], sent2_embeds_test[i]])
        

    diffs_test = []
    cosine_sims_test = []
    euclidean_dist_test = []
    l1_dist_test = []
    for i in range(len(sent1_embeds_test)):
        sent1_vec = both_list_test[i][0].reshape(1,-1)
        sent2_vec = both_list_test[i][1].reshape(1,-1)
        cosine_sims_test.append(cosine_similarity(sent1_vec, sent2_vec)[0][0])
        dif = sent1_vec - sent2_vec
        diffs_test.append(torch.squeeze(dif, dim=0))

        euclidean_dist_test.append(np.linalg.norm(sent1_vec - sent2_vec))
        l1_dist_test.append(np.linalg.norm(sent1_vec - sent2_vec, ord=1))
        
        
    for i in range(len(diffs_test)):
        diffs_test[i] = np.array(diffs_test[i])



    np.stack(diffs_train).shape
    X_train_part1 = np.stack(diffs_train)
    X_train_part2 = np.vstack([cosine_sims_train, euclidean_dist_train, l1_dist_train] ).transpose()
    X_train = np.hstack([X_train_part1, X_train_part2])

    np.stack(diffs_test).shape
    X_test_part1 = np.stack(diffs_test)
    X_test_part2 = np.vstack([cosine_sims_test, euclidean_dist_test, l1_dist_test] ).transpose()
    X_test = np.hstack([X_test_part1, X_test_part2])

    X_train = X_train_part2 #better to ignore the embeddings that lead to overfitting
    X_test = X_test_part2 #better to ignoret the embeddings that lead to overfitting
    clf = LogisticRegression(random_state=0, max_iter=2000)
    clf.fit(X_train, y_train)
    y_train_preds = clf.predict(X_train)

    print('Training accuracy: ' + str(accuracy_score(y_train, y_train_preds)))


    clf = LogisticRegression(random_state=0, max_iter=2000)
    clf.fit(X_train, y_train) #********************** again need to fit on training
    y_test_preds = clf.predict(X_test)

    print('Testing accuracy: ' + str(accuracy_score(y_test, y_test_preds)))

    y_outfile_format = []
    for i in range(len(y_test_preds)):
        if y_test_preds[i] == 1:
            y_outfile_format.append('T')
        elif y_test_preds[i] == 0:
            y_outfile_format.append('F')
        else:
            print('error')
            
    with open(args.out_file, 'w') as f:
        for i in y_outfile_format:
            f.write(i)
            f.write('\n')
    # ('out_file.txt', y_outfile_format)

    # y_errors = y_test - y_testing_preds

    ####################################################################################
    ########################################################################################
    ###################################################################################


