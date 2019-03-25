import os
from data_loader import load_word_dataset
import random
import numpy as np
import sys


def load_dataset(f_name):
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
        else:
            seg_flags.append(0)
        sent_x.append(char)
        sent_y.append(label)
        x_char.append(char)
        y_char.append(label)
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


def create_noisy_dataset_word(all_sent_x_word, all_sent_y_word, delete_spc_prob, insert_spc_prob ):
    corrupt_x = []
    corrupt_y = []
    count_dels = 0
    count_ins = 0
    for i, (sentence_x, sentence_y) in enumerate( zip(all_sent_x_word, all_sent_y_word)):
        corrupt_sent_x = []
        corrupt_sent_y = []
        j = 0
        while j < len(sentence_x):
            word = sentence_x[j]
            label = sentence_y[j]
            
            #If current word is space just append it to the output
            if word == ' ':
                corrupt_sent_x.append(word)
                corrupt_sent_y.append(label)
                j += 1
                continue
            if j < len(sentence_x) - 1:
                next_word = sentence_x[j+1]
                
            if next_word ==  ' ':
                if j+2 < len(sentence_x):
                    word_aft_spc = sentence_x[j+2]
                    label_aft_spc = sentence_y[j+2]
                    
                    sample_del = np.random.uniform()
                    if sample_del < delete_spc_prob:
                        count_dels += 1
                        merged_token = word + word_aft_spc
                        merged_label = random.choice([label,label_aft_spc ] )
                        
                        corrupt_sent_x.append(merged_token)
                        corrupt_sent_y.append(merged_label)
                        j = j + 3
                        continue
                    
            j += 1
            corrupt_token = ''
            for char in word[:-1]:
                sample_ins = np.random.uniform()
                corrupt_token += char
                if sample_ins < insert_spc_prob:
                    count_ins += 1
                    corrupt_sent_x.append(corrupt_token)
                    corrupt_sent_y.append(label)
                    corrupt_sent_x.append(' ')
                    corrupt_sent_y.append('SPACE')
                    corrupt_token = ''
            corrupt_token += word[-1]
            
            corrupt_sent_x.append(corrupt_token)
            corrupt_sent_y.append(label)
            
        corrupt_x.append(corrupt_sent_x)
        corrupt_y.append(corrupt_sent_y)
        
    print ('Dels', count_dels)
    print ('Ins', count_ins)
    print ('Dels/Ins ratio', count_dels / count_ins)
    return corrupt_x, corrupt_y


def remove_space(x_data, y_data):
    x_no_spc = [] 
    y_no_spc = [] 
    for (sentence_x, sentence_y) in zip(x_data, y_data):
        new_sent_x = []
        new_sent_y = []
        for (word,label) in zip(sentence_x, sentence_y):
            if label != 'SPACE':
                new_sent_x.append(word)
                new_sent_y.append(label)
                
        x_no_spc.append(new_sent_x)
        y_no_spc.append(new_sent_y)
    return x_no_spc, y_no_spc


def remove_file(f_name):
    try:
        os.remove(f_name)
    except OSError:
        pass


def save_dataset_format_train(f_name, x_data, y_data):         
    remove_file(f_name)    
    with open(f_name, 'w') as f:
        for _, (sentence, sent_labels) in enumerate(zip(x_data, y_data)):
            counter = 1
            for _, (word, label) in enumerate( zip(sentence, sent_labels)):
                    f.write(str(counter) + '\t' + word +  '\t' + label +'\n' )
                    counter +=1
            f.write('\n') 


def save_dataset_format_test(f_name, x_data):        
    remove_file(f_name)    
    with open(f_name, 'w') as f:
        for _, sentence in enumerate(x_data,):
            for _, word in enumerate( sentence ):
                    f.write( word +'\n' )
            f.write('\n') 


def save_train_dataset_char_corrupt(f_name, x_data, y_data):         
    remove_file(f_name)    
    with open(f_name, 'w') as f:
        for _, (sentence, labels) in enumerate(zip(x_data, y_data)):
            for _, (word, label) in enumerate( zip(sentence, labels)):
                for j, char in enumerate(word):
                        f.write(char + '\t' + label + '\n' )
            f.write('\n')       


def load_dataset_char_corrupt(f_name):
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
            x = data[0]
            sent_x.append(x)
            label = data[1]
            sent_y.append(label)
            
    return all_x, all_labels


if __name__ == '__main__':
    random.seed(0)
    
    f_name_train = 'data/words/en-ud-train1.2.conllu'
    f_name_val = 'data/words/en-ud-dev1.2.conllu'
    f_name_test =  'data/words/en-ud-test1.2.conllu'
    
    
    all_sent_x_word_train, all_sent_y_word_train, _, _ = load_word_dataset(f_name_train)
    all_sent_x_word_val, all_sent_y_word_val, _, _ = load_word_dataset(f_name_val)
    all_sent_x_word_test, all_sent_y_word_test, _, _ = load_word_dataset(f_name_test)
    
    delete_spc_prob = 0.1
    insert_spc_prob =  0.05
    corrupt_x_train, corrupt_y_train = create_noisy_dataset_word(all_sent_x_word_train, all_sent_y_word_train,
                                                                delete_spc_prob, insert_spc_prob )
    
    corrupt_x_val, corrupt_y_val = create_noisy_dataset_word(all_sent_x_word_val, all_sent_y_word_val,
                                                                delete_spc_prob, insert_spc_prob )
    
    
    corrupt_x_test, corrupt_y_test = create_noisy_dataset_word(all_sent_x_word_test, all_sent_y_word_test,
                                                                delete_spc_prob, insert_spc_prob )
    
    #Used for marmot clean text score
    clean_x_no_spc_train, clean_y_no_spc_train = remove_space(all_sent_x_word_train, all_sent_y_word_train)
    clean_x_no_spc_test, clean_y_no_spc_test = remove_space(all_sent_x_word_test, all_sent_y_word_test)
    
    #Used for marmot corrupt score
    corrupt_x_no_spc_train, corrupt_y_no_spc_train = remove_space(corrupt_x_train, corrupt_y_train)
    corrupt_x_no_spc_test, corrupt_y_no_spc_test = remove_space(corrupt_x_test, corrupt_y_test)
    
    
    #Save both clean and corrupt datasets in marmot expected format
    CORRUPT = False
    if CORRUPT == True:
        f_name_word_train_corrupt = 'data/words/corrupt/en-ud-train1.2_cor' + str(delete_spc_prob)+ '-' +str(insert_spc_prob) +  '.conllu'
        f_name_word_test_corrupt = 'data/words/corrupt/en-ud-val1.2_cor' + str(delete_spc_prob)+ '-' +str(insert_spc_prob) +  '.conllu'
        f_name_test_input_corrupt = 'data/words/corrupt/en-ud-test1.2_cor' + str(delete_spc_prob)+ '-' +str(insert_spc_prob) +  '.conllu'
        
        save_dataset_format_train(f_name_word_train_corrupt, corrupt_x_no_spc_train, corrupt_y_no_spc_train )
  #      save_dataset_format_train(f_name_word_test_corrupt, corrupt_x_no_spc_test, corrupt_y_no_spc_test )
        
        save_dataset_format_test(f_name_test_input_corrupt, corrupt_x_no_spc_test) 
        
        #These files are for our char model
        f_name_char_train_corrupt = 'data/char/corrupt/en-ud-train1.2_cor'  + str(delete_spc_prob)+ '-' +str(insert_spc_prob) +  '.conllu'
        f_name_char_val_corrupt = 'data/char/corrupt/en-ud-val1.2_cor'  + str(delete_spc_prob)+ '-' +str(insert_spc_prob) +  '.conllu'
        f_name_char_test_corrupt = 'data/char/corrupt/en-ud-test1.2_cor'  + str(delete_spc_prob)+ '-' +str(insert_spc_prob) +  '.conllu'
        
        
        
        save_train_dataset_char_corrupt(f_name_char_train_corrupt,  corrupt_x_train, corrupt_y_train)
        save_train_dataset_char_corrupt(f_name_char_val_corrupt,  corrupt_x_val, corrupt_y_val)
        save_train_dataset_char_corrupt(f_name_char_test_corrupt,  corrupt_x_test, corrupt_y_test)
        
    else:
        f_name_word_train_clean = 'data/words/clean/en-ud-train1.2_marmot_clean.txt'
        f_name_word_test_clean = 'data/words/clean/en-ud-test1.2_marmot_clean_labels.txt'
        f_name_test_input_clean = 'data/words/clean/en-ud-test1.2_marmot_clean_test_input.txt'
        
        
        save_dataset_format_train(f_name_word_train_clean, clean_x_no_spc_train, clean_y_no_spc_train )
    #    save_dataset_format_train(f_name_word_test_clean, clean_x_no_spc_test, clean_y_no_spc_test )
        
        save_dataset_format_test(f_name_test_input_clean, clean_x_no_spc_test)  
    
    
    
