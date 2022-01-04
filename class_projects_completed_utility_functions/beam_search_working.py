# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:33:46 2022

@author: 16028
"""


def image_beam_decoder(_word_to_id, _id_to_word, _caption_model, _max_len, _device, n, _enc_image):
  def words_to_ids(sequence):
    return [_word_to_id[word] for word in sequence]
  def ids_to_words(numer_sequence):
    return [_id_to_word[id] for id in numer_sequence]

  word_seq = ['<START>']
  start_seq_num = _word_to_id['<START>']
  seq = [start_seq_num]
  N = n
  beam_width = N



  numeric_sequence = [_word_to_id[curr_word] for curr_word in word_seq]
  tensor_sequence_fixed = torch.tensor(numeric_sequence, dtype=torch.int32, device=_device).unsqueeze(dim=0)
  _enc_image_fixed = _enc_image.unsqueeze(dim=0)
  # print('made itttt')

  logits_list = _caption_model(_enc_image_fixed.to(_caption_model.device), tensor_sequence_fixed.to(_caption_model.device))
  top_n_ids = torch.topk(logits_list,N).indices
  top_n_probs = torch.topk(logits_list,N).values

  top_n_ids = top_n_ids.squeeze(dim=0).tolist()
  # print(top_n_ids)
  sequences = [numeric_sequence + [id] for id in top_n_ids] #check that this is working
  # print(sequences)

  not_all_are_done = False
  final_candidates = []
  final_candidates_prob_scores = []

  num_done = 0
# sequences_dict = [seq, True]
  while (num_done < beam_width):

    top_n_id_array = torch.zeros((N,beam_width), dtype = torch.int32)
    top_n_probs_array = torch.zeros((N,beam_width), dtype = torch.float64)
    for i in range(len(sequences)):
      tensor_sequence_fixed = torch.tensor(sequences[i], dtype=torch.int32, device=_device).unsqueeze(dim=0)
      _enc_image_fixed = _enc_image.unsqueeze(dim=0)
      logits_list = _caption_model(_enc_image_fixed.to(_caption_model.device), tensor_sequence_fixed.to(_caption_model.device))
      # top_n_id_array[i] = np.array(torch.topk(logits_list,beam_width).indices.cpu(), dtype=np.int32)
      top_n_id_array[i] = torch.topk(logits_list, beam_width).indices
      top_n_probs_array[i] = torch.topk(logits_list, beam_width).values

      # top_n_probs_array[i] = np.array(torch.topk(logits_list,beam_width).values.cpu())
      
      # top_n_probs_array[i] = torch.topk(logits_list,beam_width).values.detach().cpu().numpy()
      ####


    # print(top_n_id_array)
    # print('heyooo')
    v, i = torch.topk(top_n_probs_array.flatten(), N)

    # print(np.array(np.unravel_index(i.numpy(), top_n_probs_array.shape)).T)

    row_col = np.array(np.unravel_index(i.numpy(), top_n_probs_array.shape)).T
    # print(row_col)
    # idx = np.argsort(top_n_probs_array.ravel())[-N:][::-1]
    # topN_val = top_n_probs_array.ravel()[idx]
    # row_col = np.c_[np.unravel_index(idx, top_n_probs_array.shape)]
    # # https://stackoverflow.com/questions/46554647/returning-the-n-largest-values-indices-in-a-multidimensional-array-can-find-so
    # citing the above for above lines picking out the top N across 2d array

    topns = torch.zeros((N),dtype=torch.int32 )
    old_sequences = sequences.copy()
    sequences = []
    # print(top_n_id_array)
    # print('the shape:')
    # print(top_n_id_array.shape)
    for i in range(N):
        # print('before the hey')
        topns[i] = top_n_id_array[row_col[i,0],row_col[i,1]]

        if _id_to_word[topns[i].item()] != '<END>':
          sequences.append(old_sequences[i] + [topns[i].item()])
        else:
          # print('one done!')
          # print(old_sequences[i])
          # print(_id_to_word[topns[i].item()])
          old_sequences[i].append(topns[i].item())
 

          # print(ids_to_words(old_sequences[i] )) #why did i need to cast this?
          N = N - 1
          num_done = num_done + 1
          final_candidates.append(old_sequences[i])
          final_candidates_prob_scores.append(top_n_probs_array[row_col[i,0],row_col[i,1]] )
  myguy = np.array(final_candidates_prob_scores).argmax()
  print(myguy)
  best_seq = ids_to_words(final_candidates[myguy])
  # best_seq = final_candidates[final_candidates_prob_scores.argmax()]
  return best_seq

  
# image_beam_decoder(word_to_id, id_to_word, caption_model, MAX_LEN, device, 5, enc_dev[100])