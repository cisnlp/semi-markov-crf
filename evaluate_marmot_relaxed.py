import random
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from collections import Counter
import itertools


def load_word_dataset(f_name):
    "Loads word level Universal Dependencies dataset"
    with open(f_name) as f:
        contents = f.readlines()   
    x_word = []
    y_word = []
    
    sent_x = []
    sent_y = []
    all_sent_x = []
    all_sent_y = []
    
    for i, line in enumerate(contents):
    
        data = line.split()
        if not data:
           all_sent_x.append(sent_x)
           all_sent_y.append(sent_y)
           sent_x = []
           sent_y = [] 
           continue
        word = data[1]
        label = data[3]

            
        x_word.append(word)
        y_word.append(label)

        sent_x.append(word)
        sent_y.append(label)
        
    return all_sent_x, all_sent_y, x_word, y_word 
   
def load_char_dataset(f_name):
    
    contents = []
    with open(f_name) as f:
            contents = f.readlines()
    x_char = []
    y_char = []
    
    sent_x = []
    sent_y = []
    all_sent_x = []
    all_sent_y = []
    all_sent_segs = []
    seg_flags = []
    skip = False
    for i, line in enumerate(contents[:-1]):
        if (skip):
            skip = False
            continue
        data = line.split('\t')
        
        next_data = contents[i+1].split()
        char, label, flag = data
        flag = flag.strip('\n')
        if (label != 'SPACE'):
            seg_flags.append(int(flag))
        x_char.append(char)
        y_char.append(label)
        sent_x.append(char)
        sent_y.append(label)
        if not next_data:
           all_sent_x.append(sent_x)
           all_sent_y.append(sent_y)
           all_sent_segs.append(seg_flags)
           sent_x = []
           sent_y = []
           seg_flags = []
           skip = True
           continue

    return all_sent_x, all_sent_y, all_sent_segs, x_char, y_char 
 
    
def load_marmot_dataset(f_name):
    with open(f_name, 'r') as f:
        all_x = []
        sent_x = []
        all_labels = []
        sent_y = []
        for line in f:
            data = line.split('\t')
            if data[0] == '\n':
                all_x.append(sent_x)
                sent_x = []
                all_labels.append(sent_y)
                sent_y = []
                continue
            x = data[1]
            sent_x.append(x)
            label = data[2].rstrip()
            sent_y.append(label)
            
    return all_x, all_labels    


def load_predictions(f_pred):
    with open(f_pred, 'r') as f:
        all_x = []
        sent_x = []
        all_labels = []
        sent_y = []
        for line in f:
            data = line.split('\t')
            if data[0] == '\n':
                all_x.append(sent_x)
                sent_x = []
                all_labels.append(sent_y)
                sent_y = []
                continue
            x = data[1]
            sent_x.append(x)
            label = data[5]
            sent_y.append(label)
            
    return all_x, all_labels   


def get_wrong_label(clean_tag, labels_str):
    
    wrong_label = random.choice(labels_str)
    while wrong_label == clean_tag:
        wrong_label = random.choice(labels_str)
    
    return wrong_label


def calculate_merged(reconstr_sent,cor_pred, clean_sent, clean_idx, merged_tok_num):
    
    for i in range(merged_tok_num):
        if cor_pred == clean_sent[clean_idx+i]:
            reconstr_sent.append(cor_pred)
        else:
            wrong_label = get_wrong_label(clean_sent[clean_idx+i], labels_str)
            reconstr_sent.append(wrong_label)  
            
    return reconstr_sent    


if __name__ == '__main__': 
    f_clean_test = 'data/words/clean/en-ud-test1.2_marmot_clean_labels.txt'
    _ , clean_original = load_marmot_dataset(f_clean_test)
    
    f_clean_preds = 'data/words/clean/marmot_predictions_clean.txt'
    all_sent_x_test,  clean_preds = load_predictions(f_clean_preds)
    
    
    clean_labels_flat = list(itertools.chain.from_iterable(clean_original))
    clean_preds_flat = list(itertools.chain.from_iterable(clean_preds))
    
    word_acc = accuracy_score( clean_labels_flat, clean_preds_flat) 
    print ("NON UDPIPE-TOKENIZED ACCURACY (Table 1): \n", word_acc)
    print("\n")
    
    
    labels_str = list(set(clean_labels_flat))
    f_name_test =  'data/char/en1.2/en-ud-test1.2.conllu'
    _, _, seg_flags_test, _, _ = load_char_dataset(f_name_test) 
    
    f_marmot_preds = ['data/words/udpipe_files/ud_clean_tokenized_marmot_output.txt',
                      'data/words/udpipe_files/ud_corr_low_tokenized_marmot_output.txt',
                      'data/words/udpipe_files/ud_corr_med_tokenized_marmot_output.txt',
                      'data/words/udpipe_files/ud_corr_high_tokenized_marmot_output.txt']
    
    corruption_level = ['CLEAN', 'LOW', 'MED', 'HIGH']
    
    print ("Marmot acc POS (Table 4)")
    
    #Outer for is about each corruption level
    for f_marmot_pred, corr_level in zip(f_marmot_preds,corruption_level):
        corrupt_x , corrupt_preds = load_predictions(f_marmot_pred)
        
        #curr_flags is a binary flag used to indicate the number
        #of clean tokens on each corrupted token pass
        reconstr_all = []
        for i, sentence in enumerate(corrupt_x):
            flag_idx = 0
            clean_idx = 0
            reconstr_sent = []
            temp_token_label = []
            clean_sent = clean_original[i]
            corrupt_sent = corrupt_preds[i]
            segmented_token = False
            
            
            for j, token in enumerate(sentence):
                curr_flags = seg_flags_test[i][flag_idx : flag_idx+ len(token)]
                flag_idx += len(token)
                
                #Case where we have a splitted token e.g  Wo-nd-er-ful...
                if sum(curr_flags) == 0:
                    temp_token_label.append( corrupt_sent[j])
                    segmented_token = True
                    continue
                else:
                    #Case where an end of a splited token from previously is detected e.g Wo-nder-ful
                    if segmented_token == True:
                        temp_token_label.append( corrupt_sent[j])
                        if ( clean_sent[clean_idx] in temp_token_label):
                            reconstr_sent.append(clean_sent[clean_idx])
                        else: 
                            wrong_label = get_wrong_label(clean_sent[clean_idx], labels_str)
                            reconstr_sent.append(wrong_label)
        
                        clean_idx += 1
                        temp_token_label = []
                        segmented_token = False
                        
                        #Subcase where we have a second half split token - full token merge e.g Wo-nder-fulworld
                        if sum(curr_flags) == 2:
                            reconstr_sent.append(corrupt_sent[j])
                            clean_idx += 1
                            continue
                        #Subcase where there are more merges after a splitted token e.g Wo-nder-fulworldtoday
                        if sum(curr_flags) > 2:
                           reconstr_sent = calculate_merged(reconstr_sent, corrupt_sent[j],
                                                            clean_sent, clean_idx, sum(curr_flags)-1)
                           clean_idx += sum(curr_flags)-1
                           continue
                    else:
                        #Case where we have a correct tokenization
                        if sum(curr_flags) == 1:
                            reconstr_sent.append(corrupt_sent[j])
                            clean_idx += 1
                            
                        #Case where we have a merging of two or more tokens e.g Don'tgothere
                        elif sum(curr_flags) > 1:
                            reconstr_sent = calculate_merged(reconstr_sent, corrupt_sent[j],
                                                             clean_sent, clean_idx, sum(curr_flags) )
                            clean_idx += sum(curr_flags)
                            
    
            reconstr_all.append(reconstr_sent)   
    
        clean_labels_flat = list(itertools.chain.from_iterable(clean_original))
        corrupt_preds_flat = list(itertools.chain.from_iterable(reconstr_all))
    
        print('=======================')
        print('Corruption level', corr_level)
        word_acc_corr = accuracy_score( clean_labels_flat, corrupt_preds_flat)
        print (word_acc_corr)
        print('=======================\n')
    
