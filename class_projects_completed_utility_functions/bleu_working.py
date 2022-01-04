# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:35:19 2022

@author: 16028
"""
import numpy as np
import string #*** ADDED THIS IMPORT
import collections#*** added this import
from collections import defaultdict

def get_ngrams(candidate_sentence, N ):

  # N = 4

  all_candidate_ngrams = []
  for current_word_index in range(len(candidate_sentence)):
    for j in range(N): #j tells us what kind of ngram we're doing 
      current_ngram = []
      if j + current_word_index > len(candidate_sentence) - 1:
        continue
      for k in range(0,j+1): #double ranges was messsing things up
        if ( (current_word_index + k) < len(candidate_sentence) ):
          current_ngram.append(candidate_sentence[current_word_index + k])

      all_candidate_ngrams.append(current_ngram)
  # print(all_candidate_ngrams)
  nested_list_of_tuples = [tuple(l) for l in all_candidate_ngrams] #citing https://stackoverflow.com/questions/18938276/how-to-convert-nested-list-of-lists-into-a-list-of-tuples-in-python-3-3
  # tuples_1 = [entry for entry in nested_list_of_tuples if len(entry) == 1]
  # tuples_2 = [entry for entry in nested_list_of_tuples if len(entry) == 2]
  # tuples_3 = [entry for entry in nested_list_of_tuples if len(entry) == 3]
  # tuples_4 = [entry for entry in nested_list_of_tuples if len(entry) == 4]
  tuples_N = [entry for entry in nested_list_of_tuples if len(entry) == N]
  # print(tuples_1)

  # mytuples_list_separated = [tuples_1,tuples_2,tuples_3,tuples_4]
  
  # return mytuples_list_separated
  return tuples_N

def give_pn_dict(_candidate, _references, _N):


  ngram_tups = get_ngrams(_candidate, _N)
  # print(nested_list_of_tuples)


  ref_ngrams_list = []
  for ref_sentence in ref_list:
    ref_ngrams_list.append(get_ngrams(ref_sentence,_N)) #ref_ngrams_list is a list of lists of tuples lists

  candidate_ngram_grouping = ngram_tups #so this should get the list of tuples corresponding to say, 2-grams
  candidate_counts = collections.Counter(candidate_ngram_grouping) 
  # print(candidate_counts)

  ref_counts = {}
  for individual_ngram in candidate_ngram_grouping: #for each ngram in the candidate sentence
    overall_max = 0
    for current_ref_ngram in ref_ngrams_list: #so for each sentence in the list of references--each individual 'ngram_group' is a list of 4 items (aka it contains 4 lists of tuples)   #so this should get the list of tuples corresponding to say, 2-grams (using the outer loop index i)
      occurrences = collections.Counter(current_ref_ngram)[individual_ngram] #access the frequency count of this particular ngram #what if its not there?
      if occurrences > overall_max:
        overall_max = occurrences
    ref_counts[individual_ngram] = overall_max #thus we've got the maximum frequency count among all the reference sentences
  # print(str(i+1) + '-gram reference counts')

  total_candidate_number = len(candidate_ngram_grouping) #length for the particular n-gram tuples list

  # if np.array([total_candidate_number_list]).sum() == 0:
  # if total_candidate_number == 0:
  #   if _N > 0:
  #     return give_pn_dict(_candidate, _references, _N-1)

  final_counts = {}
  total_modified_ngram_precision = 0
  for k,v in candidate_counts.items(): #for each tuple in this particular ngram set
    total_modified_ngram_precision += min(candidate_counts[k], ref_counts[k])/total_candidate_number #we divide by the total number of candidate tuples

  return total_modified_ngram_precision


def bleu(_candidate, _references, _weights):
  #math.inf
  # print(_candidate)

  length_diffs = [np.inf] * len(_references) #just using a ridiculously large number to ensure it doesn't mess up our min later
  length_diffs_abs_value = [np.inf] * len(_references)

  candidate_len = len(_candidate)
  min_distance = np.inf
  r = _references[0]
  for i in range(len(_references)): #for each reference sentence

    current_ref_length = len(_references[i])
    current_distance = np.abs(current_ref_length - candidate_len)
    if current_distance < min_distance: #this also works i just have two ways of finding
      r = len(_references[i])
    length_diffs[i] = current_ref_length - candidate_len
    length_diffs_abs_value[i] = np.abs(length_diffs[i])

  length_diffs_abs_value = np.array(length_diffs_abs_value)
  # min_distance = length_diffs_abs_value.min()
  min_sentence = _references[length_diffs_abs_value.argmin()]
  c = candidate_len
  r = len(min_sentence)
  if c > r:
    brev_penalty = 1
  else:
    brev_penalty = np.exp(1-(r/c))
  N = 4
  # pn_dict = give_pn_dict(_candidate, _references, N)
  total_sum = 0
  pn_dict = {}
  weighted_sum = 0
  weights = np.array(_weights)
  # print(weights.shape[0])
  nonzero = 0
  for idx in range(1,N+1):
    pn = give_pn_dict(_candidate, _references, idx)
    if pn != 0:
      # weighted_sum += weights[idx-1] * np.log(pn)
      nonzero += 1
      pn_dict[idx] = pn

  if nonzero == 0:
    return 0

  if nonzero != weights.shape[0]:
    weights = np.ones(nonzero) * 1/nonzero
    # print(weights)

  pn_values = list(pn_dict.values())
  pn_values = np.array(np.log(pn_values))
  weighted_sum = (weights*pn_values).sum() #doublecheck
  total_val = brev_penalty*np.exp(weighted_sum)
  return total_val

  
    