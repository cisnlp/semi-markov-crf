import torch 
import numpy as np
import os
import random
import re

random.seed(0)
seed = 0
torch.manual_seed(seed)

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
    
        data = line
        if data[0] == '\n':
           all_sent_x.append(sent_x)
           all_sent_y.append(sent_y)
           sent_x = []
           sent_y = [] 
           continue
       
        if data[0] == '#':
            continue
        line = line.rstrip()
        data = re.split(r'\t', line)
        
        word = data[1]
        label = data[3]

            
        x_word.append(word)
        y_word.append(label)

        sent_x.append(word)
        sent_y.append(label)
        if (data[-1] != 'SpaceAfter=No'):
            sent_x.append(" ")
            sent_y.append('SPACE')
        
    return all_sent_x, all_sent_y, x_word, y_word  



def remove_file(f_name):
    try:
        os.remove(f_name)
    except OSError:
        pass
    
    

def save_char_file(all_sent_x, all_sent_y, f_name, ):
    """ Saves the data in f_name with format char \t label \t end-of-word-flag \n """
    with open(f_name, 'w+') as f:
        for _, (sentence, sent_labels) in enumerate(zip(all_sent_x, all_sent_y)):
            for _, (word, label) in enumerate( zip(sentence, sent_labels)):
                for j, char in enumerate(word):
                    
                    #For tokens with space
                    to_write_label = label
                    if char == ' ':
                        to_write_label = 'SPACE'
                        
                    if j == len(word)-1:
                        f.write(char + '\t' + to_write_label +'\t' + '1'+ '\n' )
                    else: 
                        f.write(char + '\t' + to_write_label +'\t' + '0' + '\n')
            f.write('\n')

def create_char_dataset(dir_name,f_name_char_train, f_name_char_val, f_name_char_test):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    all_sent_x_train, all_sent_y_train, x_word, y_word = load_word_dataset(f_name_word_train)
    all_sent_x_val , all_sent_y_val, _, _ = load_word_dataset(f_name_word_val)
    all_sent_x_test , all_sent_y_test, _, _ = load_word_dataset(f_name_word_test)
    
    #Remove previous files if they exist
    remove_file(f_name_char_train)            
    remove_file(f_name_char_val)            
    remove_file(f_name_char_test)           
    
    save_char_file(all_sent_x_train,all_sent_y_train, f_name_char_train)
    save_char_file(all_sent_x_val,all_sent_y_val, f_name_char_val)
    save_char_file(all_sent_x_test,all_sent_y_test, f_name_char_test)

def load_char_dataset(f_name):
    
    """Loads the sentences in character format
    from a preprocessed char file along with the word boundary"""
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
        #Skip if blank line
        if (skip):
            skip = False
            continue
        data = line.split('\t')
        next_data = contents[i+1].split()
        char, label, flag = data
        flag = flag.strip('\n')
        
        #Dont append the space flags since they will be removed after encoding
        if (label != 'SPACE'):
            seg_flags.append(int(flag))
        sent_x.append(char)
        sent_y.append(label)
        x_char.append(char)
        y_char.append(label)
        
        #Append the sentence when a blank line is spotted
        if not next_data:
           all_sent_x.append(sent_x)
           all_sent_y.append(sent_y)
           all_sent_segs.append(seg_flags)
           sent_x = []
           sent_y = []
           seg_flags = []
           #Next line is blank, skip it
           skip = True
           continue

    return all_sent_x, all_sent_y, all_sent_segs, x_char, y_char   

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def batchify_x(data, batch_size, batch_len_list, char_enc_size):
    """Takes a list of data_num x len x input_dim
        returns a list of batch_size x max_len_of_batch x input dim  tensors """
        
    batched_data = []
    
    for i, batch_no_pad in enumerate(chunks(data, batch_size)):
        
        max_sent_len = max(batch_len_list[i])
        for i, sent in enumerate(batch_no_pad):
            batch_no_pad[i] =  np.vstack( (sent, np.zeros((max_sent_len - len(sent), char_enc_size), dtype=np.float32)  ) )
        
        batch_no_pad = np.array(batch_no_pad)
        batch_no_pad = batch_no_pad.transpose((1,0,2))
        batch_no_pad = torch.from_numpy(batch_no_pad)

       
       
        batched_data.append(batch_no_pad)
                                
    return batched_data


def batchify_y(data, batch_size, batch_len_list, char_enc_size):
    """Takes a list of data_size x sent_len
        returns a list of batch_size x max_len_of_batch  tensors """
    
    batched_data = []
    
    for i, batch_no_pad in enumerate(chunks(data, batch_size)):
        
        max_sent_len = max(batch_len_list[i])

        for i, sent in enumerate(batch_no_pad):
            sent.extend([np.uint8(-1)] * (max_sent_len - len(sent)))  
        

        batch_no_pad = np.array(batch_no_pad)
        batch_no_pad = batch_no_pad.transpose()
        batch_no_pad = torch.from_numpy(batch_no_pad)
       
        batched_data.append(batch_no_pad)
                                
    return batched_data

def batch_len_list(len_list, batch_size):
    """ Creates a list of the length of each minibatch"""
    
    if (len(len_list) % batch_size != 0):
        batch_num = len(len_list) // batch_size + 1
    else:
        batch_num = len(len_list) // batch_size
        
    batched_len_list = []
    gen = chunks(len_list, batch_size)
    for i in range(batch_num):
        batched_len_list.append( next(gen)  )
    return batched_len_list



# =============================================================================
# def max_pad_sentences(all_sent_x_one_hot, all_sent_y, all_seg_flags, max_sent_len, char_enc_size):
#     """ Pad the length of each example to the maximum so we equal sized chunks
#         for easier processing"""    
#         
#     for i, t in enumerate (all_sent_x_one_hot):
#        all_sent_x_one_hot[i] =  np.vstack( (t, np.zeros((max_sent_len - len(t), char_enc_size), dtype=np.float32)  ))
#        
#     for t in all_sent_y:
#         t.extend([np.uint8(-1)] * (max_sent_len - len(t)))        
#     
#     for t in all_seg_flags:
#         t.extend([np.uint8(-1)] * (max_sent_len - len(t))) 
# =============================================================================
        
    return 

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
            label = data[1].rstrip()
            sent_y.append(label)
            
    return all_x, all_labels

def one_hot_encoding(all_sent_x, all_sent_y, char_to_ind, label_to_ind, char_enc_size, HAS_SPACES):
    """
    Returns the one hot encoding with the space feature
    and the int encoding of each label
    """
    all_sent_x_one_hot = []
    all_sent = list(zip(all_sent_x, all_sent_y))
    for sentence in all_sent:
        sent_inds = []
        spc_aft_char_ind = []
        spc_bef_char_ind = []
        prev_label = 'TEMP'
        oov_inds = []
        #Get character-label tuples of sentence
        sentence = list(zip(*sentence))
        for j, data in enumerate(sentence):
            char, label = data
            if (j < len(sentence)-1):
                _, next_label = sentence[j+1]
            else:
                next_label = 'TEMP'
            
            next_flag = 0
            prev_flag = 0

            if label == 'SPACE':
                prev_label = label
                continue
            if next_label == 'SPACE':
                next_flag = 1
            if prev_label == 'SPACE':
                prev_flag = 1
            if char in char_to_ind:
                ind = char_to_ind[char]
            else:
               print ('OOV CHAR', char)
               oov_inds.append(len(sent_inds))
               #Temp char, we zero it after
            #   ind = char_to_ind['$']
               ind = 0
               
            sent_inds.append(ind)
            spc_aft_char_ind.append(next_flag)
            spc_bef_char_ind.append(prev_flag)   
            prev_label = label
        
        
        one_hot_sent = np.eye(char_enc_size, dtype=np.float32)[sent_inds]
        if oov_inds:
            one_hot_sent[oov_inds, :] = 0 
            
        if HAS_SPACES:    
            one_hot_sent[:, -2] = spc_bef_char_ind
            one_hot_sent[:, -1] = spc_aft_char_ind 
            
        all_sent_x_one_hot.append(one_hot_sent)
   
    all_sent_y_int = []
    for sentence in all_sent_y:
        sent_labels = []
        for label in sentence:
           if label == 'SPACE':
               continue
           ind = label_to_ind[label]
           sent_labels.append(ind)
        all_sent_y_int.append(sent_labels) 
    
    return all_sent_x_one_hot, all_sent_y_int

def seg_mask_fix(seg_inds, max_path):
    counter = np.zeros((len(seg_inds)), dtype=np.int32)
    seg_inds_fix = []
    for b,  sent_inds in enumerate(seg_inds):
        counter = 0
        new_inds = []
        for i , flag in enumerate(sent_inds):
            path_flag = (counter >= max_path-1)
                
            mask_step = flag | path_flag
            new_inds.append(mask_step)
            counter = counter + 1
            counter = (1- mask_step)*counter*(counter < max_path)
            
        seg_inds_fix.append(new_inds)
        
    return seg_inds_fix

def encode_and_batch(all_sent_x, all_sent_y, all_seg_ind, char_to_ind, label_to_ind, batch_size, char_enc_size, max_path, HAS_SPACES=True, train=False):
    
    all_sent_x_one_hot , all_sent_y_int =  one_hot_encoding(all_sent_x, all_sent_y ,char_to_ind,
                                                            label_to_ind, char_enc_size, HAS_SPACES )
    
    
    if train:
        print ('Load training data')
        
        #Bound segments that have length bigger than the max path
        all_seg_ind = seg_mask_fix(all_seg_ind, max_path)
        #Shuffle training data
        inds = list(range(len(all_sent_x)))
        random.shuffle(inds)
        all_sent_x_one_hot = [all_sent_x_one_hot[i] for i in inds]
        all_sent_y_int = [all_sent_y_int[i] for i in inds]
        all_seg_ind = [all_seg_ind[i] for i in inds]
        

    

    len_list= [np.shape(sent)[0] for sent in all_sent_x_one_hot]
    batched_len = batch_len_list(len_list, batch_size)    
    
    x_data = batchify_x(all_sent_x_one_hot, batch_size, batched_len, char_enc_size)
    y_data = batchify_y(all_sent_y_int, batch_size, batched_len, char_enc_size)
    seg_ind = batchify_y(all_seg_ind, batch_size, batched_len, char_enc_size)
    
    return x_data, y_data, seg_ind, batched_len

if __name__ == "__main__":
# =============================================================================
#     f_name_char_train = 'data/char/en-ud-train1.2.conllu'
#     f_name_char_val = 'data/char/en-ud-dev1.2.conllu'
#     f_name_char_test = 'data/char/en-ud-test1.2.conllu'
# =============================================================================
    f_name_word_train = 'data/words/en1.2/en-ud-train1.2.conllu'
    f_name_word_val = 'data/words/en1.2/en-ud-dev1.2.conllu'
    f_name_word_test =  'data/words/en1.2/en-ud-test1.2.conllu'
    
    LANG = 'en12' 
    dir_name = 'data/char' + '/' + LANG 
    
    f_name_char_train = dir_name + '/en-ud-train1.2.conllu'
    f_name_char_val = dir_name + '/en-ud-dev1.2.conllu'
    f_name_char_test = dir_name + '/en-ud-test1.2.conllu'
    
    create_char_dataset(dir_name, f_name_char_train, f_name_char_val, f_name_char_test )
    print ('Created char dataset at {}'.format(dir_name))