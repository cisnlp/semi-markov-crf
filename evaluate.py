import torch 
import numpy as np
from torch.autograd import Variable
from collections import Counter
import itertools
from sklearn.metrics import f1_score, accuracy_score

def evaluate (net, x_data, y_data, seg_ind, batched_len_list, opt):
    net.eval()
    batch_size_eval = opt.batch_size_eval
    hidden = net.init_hidden(batch_size_eval)
    num_mini_batch = len(x_data) 
    
    #List of lists, holds the char level prediction for each sentence in the evaluation set
    all_char_paths = []
    
    #List of lists, holds the segment length prediction for each token in a sentence in the evaluation set
    all_segments = []
    
    #List of lists, holds the char level labels for each sentence in the evaluation set
    all_labels_paths = []
    
    #List of lists, holds the golden word segmentation indices, char level
    all_seg_inds = []
    
    for i in range(num_mini_batch):
        x_batch = x_data[i]
        y_batch = y_data[i]
        seg_ind_batch = seg_ind[i]
        len_list = batched_len_list[i]
        
        bs = x_batch.size(1)
        
        sorted_inds_vals = [t for t in sorted(enumerate(len_list), reverse=True, key=lambda x:x[1])]
        sorted_inds, sorted_vals = map(list, zip(*sorted_inds_vals))
        sorted_inds = np.array(sorted_inds)
        sorted_inds_t = torch.LongTensor(sorted_inds)
        
        if opt.USE_CUDA == True:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            seg_ind_batch = seg_ind_batch.cuda()
            sorted_inds_t = sorted_inds_t.cuda()
            
        
        x_batch_s = torch.index_select(x_batch, 1, sorted_inds_t )
        y_batch_s = torch.index_select(y_batch, 1, sorted_inds_t )
        seg_ind_s = torch.index_select(seg_ind_batch, 1, sorted_inds_t )
        
        x_batch_s = Variable(x_batch_s, volatile = True)
        y_batch_s = Variable(y_batch_s, volatile = True)
        seg_ind_s = Variable(seg_ind_s, volatile = True)
        

        hidden = net.init_hidden(bs)
        
        output_2d, hidden = net( x_batch_s, hidden, sorted_vals )

        pack_y = torch.nn.utils.rnn.pack_padded_sequence(y_batch_s, sorted_vals)
        unpacked_y, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack_y)
        
        
        
        sorted_vals = torch.LongTensor(sorted_vals)
        if opt.USE_CUDA == True:
            sorted_vals = sorted_vals.cuda()
        #Get batch char level predictions for each sentence (List of lists)
        tag_seqs_2d, segments = net.viterbi_decode(output_2d, sorted_vals)
        
        #Extend the all_char_paths list with the tag list of lists
        all_char_paths.extend( tag_seqs_2d )
        
        #Extend the all_char_paths list with the segment length list of lists
        all_segments.extend(segments)
        
        #Get char labels of the batch and put them in a list of lists
        sent_batch_labels = [ sublist[:sorted_vals[j]] for j, sublist in enumerate(y_batch_s.transpose(1,0).data.cpu().numpy())]
        
        #Same for indices so we can recover word level predictions
        sent_batch_seg_inds = [ sublist[:sorted_vals[j]] for j, sublist in enumerate(seg_ind_s.transpose(1,0).data.cpu().numpy())]
        
        #Extend the list of lists with the new batch list
        all_labels_paths.extend( sent_batch_labels )
        all_seg_inds.extend(sent_batch_seg_inds)
        
        
        
        print('Evaluating batch', i)
    
    new_segments = []
    
    for segment in all_segments:
        temp_seg = [(0,segment[0]-1)]
        for j, val in enumerate(segment[1:],1):
            temp_seg.append((segment[j-1], val-1))
        new_segments.append(temp_seg)
    
    #Get word level predictions for the whole evaluation set    
    F1_pos_seg, F1_tok, all_words_labels, all_words_preds = convert_to_word(all_labels_paths,
                                                                            all_char_paths,
                                                                            all_seg_inds,
                                                                            new_segments)
    
    #Flatten to calculate f1 score on one run for word level
    all_words_labels_flat = list(itertools.chain.from_iterable(all_words_labels))
    all_words_preds_flat = list(itertools.chain.from_iterable(all_words_preds))
    
    word_acc = accuracy_score(all_words_labels_flat, all_words_preds_flat)
    
    print ('F1 Score POS & Seg', F1_pos_seg)
    print ('F1 Tokenization', F1_tok)
    print ('Word level Accuracy: ' , word_acc)
    
 #   print ('Character level F1 score: ' , f1_char)
    return F1_pos_seg


def convert_to_word(all_labels_paths, all_char_paths, seg_ind_s, segments_predicted):
    
    word_2d_labels = []
    word_2d_preds = []
    count_correct_pos_seg = 0
    total_clean_tokens = 0 
    total_predicted_tokens = 0 
    count_correct_tokens = 0
    for i, (sent, sent_seg_inds) in enumerate(zip(all_labels_paths,seg_ind_s)):
        idx_list = []
        start_ind = 0
        
        for j, flag in enumerate(sent_seg_inds):
           if  flag == 1:
               word_range = (start_ind, j)
               start_ind = j+1
               idx_list.append(word_range)
     
        
        char_seg = all_char_paths[i]
        segments = [ char_seg[s:(e+1)] for s,e in idx_list]
        word_2d_preds.append( [ Counter(seg).most_common()[0][0] for seg in segments] )
        segment_lens = segments_predicted[i]
        for num, seg in enumerate(segment_lens):
            if  seg in idx_list:
                count_correct_tokens +=1 
                
                pred_seg_label = char_seg[seg[0]]
                true_seg_label = sent[seg[0]]
                
                if pred_seg_label == true_seg_label:
                    count_correct_pos_seg +=1

        total_clean_tokens += len(idx_list)
        total_predicted_tokens += len(segment_lens)
        word_2d_labels.append( [sent[s] for s, _ in idx_list])
        
    token_prec =  count_correct_tokens / total_predicted_tokens
    token_recall =  count_correct_tokens / total_clean_tokens
    F1_tok = (2 * token_prec * token_recall) / (token_prec + token_recall)
    
    pos_seg_prec = count_correct_pos_seg / total_predicted_tokens
    pos_seg_recall = count_correct_pos_seg / total_clean_tokens
    F1_pos_seg = (2 * pos_seg_prec * pos_seg_recall) / (pos_seg_prec + pos_seg_recall)
    
    
 #   print ('Tokenization recall' , token_recall)
 #   print ('Tokenization precision' , token_prec)
 #   print ('F1 Score Tokenization', F1_tok )
 #   print ('F1 Score POS & Seg', F1_pos_seg)
    return  F1_pos_seg, F1_tok, word_2d_labels,  word_2d_preds 