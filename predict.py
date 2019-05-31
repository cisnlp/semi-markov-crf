import torch 
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from data_loader import load_char_dataset
from data_loader import encode_and_batch
from model_grc import GRC
from evaluate import evaluate
import argparse
import pickle


def load_obj(name):
    with open('models/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    
   
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=False, help='Model path', default='models/net_vi.pt')
parser.add_argument('--segment_constr', type=str, required=False, help='Segment constructor grConv or SRNN', default='grConv')
parser.add_argument('--LANG', type=str, required=False, help='UD language to train', default='vi')
parser.add_argument('--max_path', type=int, required=True, help='Maximum segment length', default=23)
parser.add_argument('--batch_size_eval', type=int, required=True, help='Number of examples per mini-batch for test data', default=67)
parser.add_argument('--emb_size', type=int, required=True, help='Size of character embedding', default=60 )
parser.add_argument('--hidden_size', type=int, required=True, help='Size of LSTM final output hidden state (this is the concatenated size if bidrectional == True ', default=200 )
parser.add_argument('--num_layers', type=int, required=True, help='Number of LSTM layers', default=3)
parser.add_argument('--rnn_type', type=str, required=False, help='Type of RNN, either LSTM or GRU', default='LSTM' )
parser.add_argument('--drop', type=float, required=True, help='embedding dropout', default=0.25)
parser.add_argument('--recurrent_drop', type=float, required=True, help='dropout between RNN layers', default=0.25 )
parser.add_argument('--bidirectional', type=str, required=False, help='True if feature extractor is bidirectional', default='False' )
parser.add_argument('--USE_CUDA', type=str, required=True, help='Use GPU flag', default=1 )


arg_list =  ['--model_path', '/models/net_vi_grc.pt',
             '--LANG', 'vi',
             '--segment_constr', 'grConv',
             '--max_path', '23',
             '--batch_size_eval', '67',
             '--emb_size', '60',
             '--hidden_size', '200',
             '--num_layers', '3',
             '--rnn_type', 'LSTM',
             '--drop', '0.25',
             '--recurrent_drop', '0.25',
             '--bidirectional', 'True',
             '--USE_CUDA', 'False'
             ]
 

           
opt = parser.parse_args(arg_list)
model_path = os.getcwd() + opt.model_path

opt.bidirectional = str2bool(opt.bidirectional)


HAS_SPACES = True
if opt.LANG == 'en1.2':
    f_name_char_test = 'data/char/en1.2/en-ud-test1.2.conllu'
elif opt.LANG == 'en':
    f_name_char_test = 'data/char/en/en-ud-test.conllu'
elif opt.LANG == 'ja':
    f_name_char_test = 'data/char/ja/ja-ud-test.conllu'
elif opt.LANG == 'zh':
    f_name_char_test = 'data/char/zh/zh-ud-test.conllu'    
elif opt.LANG == 'vi':
    f_name_char_test = 'data/char/vi/vi-ud-test.conllu'

all_sent_x_test, all_sent_y_test, all_seg_ind_test, x_char_test, y_char_test =  load_char_dataset(f_name_char_test)


char_to_ind = load_obj('char_vocab_'  + opt.LANG)
label_to_ind = load_obj('labels_dict_'  + opt.LANG)

if HAS_SPACES:
    char_enc_size = len(char_to_ind) + 2
else:
    char_enc_size = len(char_to_ind)

x_test , y_test, seg_ind_test, batched_len_list_test = encode_and_batch(all_sent_x_test, all_sent_y_test,
                                                                        all_seg_ind_test, char_to_ind, label_to_ind,
                                                                        opt.batch_size_eval, char_enc_size, opt.max_path,
                                                                        HAS_SPACES, train=False)

if opt.segment_constr == 'grConv':      
    net_loaded = GRC(char_enc_size, label_to_ind, opt.rnn_type, opt.emb_size, opt.hidden_size, opt.num_layers,
                      opt.bidirectional, opt.max_path, opt.recurrent_drop, opt.drop)
elif opt.segment_constr == 'SRNN':
    net_loaded = SRNN(char_enc_size, label_to_ind, opt.rnn_type, opt.emb_size, opt.hidden_size, opt.num_layers,
                      opt.bidirectional, opt.max_path, opt.recurrent_drop, opt.drop)

net_loaded.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
train_score = evaluate(net_loaded, x_test, y_test, seg_ind_test, batched_len_list_test, opt)
