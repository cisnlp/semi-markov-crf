from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter
import re


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


def get_udpipe_bounds(udpipe_file):
    udpipe_bounds = []
    sent_bounds = []
    idx1 = 0
    sent = []
    with open(udpipe_file, 'r') as f:

        for line in f:
            if line[0] == '#':
                continue
            if line[0] == '\n':
                udpipe_bounds.append(sent_bounds)
                sent_bounds = []
                idx1 = 0
                sent = []
            else:
                splitted = re.split(r'\t+', line)
                word = splitted[1]
                tok_rng = splitted[-1]
                bounds = re.findall('\d+', tok_rng)
                start = int(bounds[0])
                end  = int(bounds[1])
                idx2 = end-start -1 + idx1
                tok_range = (idx1, idx2)
                idx1 = idx2 +1 
                sent_bounds.append(tok_range)
                sent.append(word)
                
    return udpipe_bounds

if __name__ == '__main__':
    print("Joint Tokenization and POS calculation\n")
    
    f_clean_test_conllu = 'data/words/udpipe_files/en-ud-test.conllu'
    all_sent_x_test_no_spc, clean_original , _, _ = load_word_dataset(f_clean_test_conllu)
    #Create char-level labels from word labels
    all_char_labels = []
    for sent,labels in zip(all_sent_x_test_no_spc, clean_original):
        char_labels = []
        for word,lbl in zip(sent,labels):
            char_labels.extend([lbl]*len(word))
        
        all_char_labels.append(char_labels)
    
    #Get bounds
    all_sent_bounds = []
    for sent in all_sent_x_test_no_spc:
        start_idx = 0
        sent_bounds = []
        for token in sent:
            sent_bounds.append( (start_idx, start_idx+len(token)-1) )
            start_idx += len(token)
            
        all_sent_bounds.append(sent_bounds)
    
    udpipe_files = ['data/words/udpipe_files/ud_clean_tokenized.txt',
                    'data/words/udpipe_files/ud_corr_low_tokenized.txt',
                    'data/words/udpipe_files/ud_corr_med_tokenized.txt',
                    'data/words/udpipe_files/ud_corr_high_tokenized.txt']
    
    
    marmot_files = ['data/words/udpipe_files/ud_clean_tokenized_marmot_output.txt',
                    'data/words/udpipe_files/ud_corr_low_tokenized_marmot_output.txt',
                    'data/words/udpipe_files/ud_corr_med_tokenized_marmot_output.txt',
                    'data/words/udpipe_files/ud_corr_high_tokenized_marmot_output.txt']
    
    corruption_level = ['CLEAN', 'LOW', 'MED', 'HIGH']
    
    for udpipe_f, marmot_f, corr_level in zip(udpipe_files, marmot_files, corruption_level):
        
        
        udpipe_bounds = get_udpipe_bounds(udpipe_f)
        _, marmot_predictions = load_predictions(marmot_f)
             
        total_clean_tokens = sum( len(elem_list) for elem_list in all_sent_bounds )
        total_predicted_tokens = sum( len(elem_list) for elem_list in udpipe_bounds )
        
        count_correct_tokens = 0
        count_correct_tags = 0 
        for i, tok_ranges in enumerate( udpipe_bounds ):
            clean_bounds = all_sent_bounds[i]
            orig_sent_labels = all_char_labels[i]
            marmot_sent_preds = marmot_predictions[i]
            
            for j, tok_rng in enumerate(tok_ranges):
                if tok_rng in clean_bounds:
                    count_correct_tokens += 1
                    clean_lbl = orig_sent_labels[tok_rng[0]]
                    marmot_lbl = marmot_sent_preds[j]
                    if clean_lbl == marmot_lbl:
                        count_correct_tags += 1
        
        
        token_recall = count_correct_tokens / total_clean_tokens
        token_prec = count_correct_tokens / total_predicted_tokens
        token_f1 = 2*token_recall*token_prec  / (token_recall + token_prec)
        
        pos_recall = count_correct_tags / total_clean_tokens
        pos_prec = count_correct_tags / total_predicted_tokens
        pos_f1 = 2*pos_recall*pos_prec  / (pos_recall + pos_prec)
        
        
        print('=======================')
        print('Corruption level', corr_level)
        print('F1 score', token_f1)
        print('Joint token-POS F1', pos_f1)
        print('=======================\n')
    




    

