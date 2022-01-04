# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:34:09 2022

@author: 16028
"""

def image_nucleus_decoder(_word_to_id, _id_to_word, _caption_model, _max_len, _device, p, _enc_image):
    #...
    # return []

    curr_word_string = "<START>"
    seq = ["<START>"]
    curr_sequence_length = 1
    _p = p

    while (curr_word_string != "<END>") & (curr_sequence_length < _max_len):
      numeric_sequence = [_word_to_id[curr_word] for curr_word in seq]
      # print(1)
      tensor_sequence = torch.tensor(numeric_sequence, dtype=torch.int32, device=_device)
      tensor_sequence_fixed = tensor_sequence.unsqueeze(dim=0)
      _enc_image_fixed = _enc_image.unsqueeze(dim=0)


      probs_list = _caption_model(_enc_image_fixed.to(_caption_model.device), tensor_sequence_fixed.to(_caption_model.device)) #was the syntax in forward

      probs_list = torch.softmax(probs_list, dim=1) 

      probs_list.sort(descending=True)
      probs_list = probs_list.squeeze()

      probs_list[5]
      prob_mass = 0
      idx = 0
      while prob_mass < _p:

        prob_mass += probs_list[idx].item()

        idx += 1
      new_word_set = probs_list[:idx]
      sum_of_mass = new_word_set.sum()
      new_word_set = new_word_set/sum_of_mass
      m = torch.distributions.Categorical(new_word_set)
      next_word_predicted = m.sample()  # equal probability of 0, 1, 2, 3


      curr_word_string = _id_to_word[next_word_predicted.item()]
      seq.append(curr_word_string)
      curr_sequence_length += 1
    final_numeric_sequence = [_word_to_id[curr_word] for curr_word in seq]
    
    return seq