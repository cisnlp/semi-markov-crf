import torch 
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
import time
import random
from data_loader import load_char_dataset
from data_loader import encode_and_batch
from model_grc import GRC
from model_srnn import SRNN
import argparse
from evaluate import evaluate
import pickle
import sys


seed = 0
torch.manual_seed(seed)
random.seed(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if (torch.cuda.is_available()):
    torch.cuda.set_device(0)
 
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    

def save_obj(obj, name ):
    with open( 'models/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=False, help='Model path', default='models/net.pt')
parser.add_argument('--LANG', type=str, required=False, help='UD language to train', default='vi')
parser.add_argument('--segment_constr', type=str, required=False, help='Segment constructor grConv or SRNN', default='grConv')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs', )
parser.add_argument('--max_path', type=int, required=True, help='Maximum segment length', )
parser.add_argument('--batch_size_train', type=int, required=True, help='Number of examples per mini-batch for train data', )
parser.add_argument('--batch_size_eval', type=int, required=True, help='Number of examples per mini-batch for validation data', )
parser.add_argument('--emb_size', type=int, required=True, help='Size of character embedding', default=60 )
parser.add_argument('--hidden_size', type=int, required=True, help='Size of LSTM final output hidden state (this is the concatenated size if bidrectional == True ', default=200 )
parser.add_argument('--num_layers', type=int, required=True, help='Number of LSTM layers', default=1)
parser.add_argument('--rnn_type', type=str, required=False, help='Type of RNN, either LSTM or GRU', default='LSTM' )
parser.add_argument('--bidirectional', type=str, required=False, help='True if feature extractor is bidirectional', default='False' )
parser.add_argument('--lr', type=float, required=True, help='Learning rate', default=0.001 )
parser.add_argument('--clip', type=float, required=True, help='Learning rate', default=1 )
parser.add_argument('--drop', type=float, required=True, help='embedding dropout', default=0.25 )
parser.add_argument('--recurrent_drop', type=float, required=True, help='dropout between RNN layers', default=0.25 )
parser.add_argument('--print_every', type=int, required=False, help='Print train loss after', default=1 )
parser.add_argument('--epoch_append', type=int, required=False, help='Calculate acc on validation every epoch_append epochs', default=1 )
parser.add_argument('--anneal_lr_every', type=int, required=False, help='Deacrease learning rate after anneal_lr_every epochs', default=10 )
parser.add_argument('--epoch_break', type=int, required=False, help='Stop training if validation accuracy hasnt increased after epoch_break consecutive epochs', default=20 )
parser.add_argument('--USE_CUDA', type=str, required=True, help='Use GPU flag', default=1 )


arg_list =  ['--model_path', '/models/net_vi_srnn_new.pt',
             '--LANG', 'vi',
             '--segment_constr', 'SRNN',
             '--num_epochs', '200',
             '--max_path', '23',
             '--batch_size_train', '20',
             '--batch_size_eval', '26',
             '--emb_size', '60',
             '--hidden_size', '200',
             '--num_layers', '3',
             '--rnn_type', 'LSTM',
             '--bidirectional', 'True',
             '--lr', '0.001',
             '--clip', '1.0',
             '--drop', '0.25',
             '--recurrent_drop', '0.25',
             '--print_every', '1',
             '--epoch_append', '1',
             '--anneal_lr_every', '10',
             '--epoch_break', '20',
             '--USE_CUDA', 'False'
             ]
 
            
opt = parser.parse_args(arg_list)
model_path = os.getcwd() + opt.model_path

opt.bidirectional = str2bool(opt.bidirectional)
opt.USE_CUDA = str2bool(opt.USE_CUDA)


HAS_SPACES = True
if opt.LANG == 'en1.2':
    f_name_char_train = 'data/char/en1.2/en-ud-train1.2.conllu'
    f_name_char_val = 'data/char/en1.2/en-ud-dev1.2.conllu'
elif opt.LANG == 'vi':
    f_name_char_train = 'data/char/vi/vi-ud-train.conllu'
    f_name_char_val = 'data/char/vi/vi-ud-dev.conllu'

all_sent_x_train, all_sent_y_train, all_seg_ind_train, x_char_train, y_char_train =  load_char_dataset(f_name_char_train)
 
all_sent_x_val, all_sent_y_val, all_seg_ind_val, x_char_val, y_char_val =  load_char_dataset(f_name_char_val)  


#Create one hot encoding dictionary
symbols = sorted(list(set(x_char_train)))
symbols.remove(' ')
sorted_symbols = symbols
labels = sorted(list(set(y_char_train)))
labels.remove('SPACE')


#Create labels to integers dictionary
labels_int = range(len(labels)) 
label_to_ind = dict( zip(labels,labels_int))
label_to_ind['START'] = len(label_to_ind)
label_to_ind['STOP'] = len(label_to_ind)

char_to_ind = {v:k for k,v in enumerate(sorted_symbols)}

num_classes = len(label_to_ind)


save_obj(char_to_ind, 'char_vocab_' + opt.LANG)
save_obj(label_to_ind, 'labels_dict_'+ opt.LANG )

if HAS_SPACES:
    char_enc_size = len(char_to_ind) + 2
else:
    char_enc_size = len(char_to_ind) 

x_train , y_train, seg_ind_train, batched_len_list_train = encode_and_batch(all_sent_x_train, all_sent_y_train,
                                                                        all_seg_ind_train, char_to_ind, label_to_ind,
                                                                        opt.batch_size_train, char_enc_size, opt.max_path,
                                                                        HAS_SPACES, train=True)

x_val , y_val, seg_ind_val, batched_len_list_val = encode_and_batch(all_sent_x_val, all_sent_y_val,
                                                                        all_seg_ind_val, char_to_ind, label_to_ind,
                                                                        opt.batch_size_eval, char_enc_size, opt.max_path,
                                                                        HAS_SPACES, train=False)


if opt.segment_constr == 'grConv':      
    net = GRC(char_enc_size, label_to_ind, opt.rnn_type, opt.emb_size, opt.hidden_size, opt.num_layers,
                      opt.bidirectional, opt.max_path, opt.recurrent_drop, opt.drop)
elif opt.segment_constr == 'SRNN':
    net = SRNN(char_enc_size, label_to_ind, opt.rnn_type, opt.emb_size, opt.hidden_size, opt.num_layers,
                      opt.bidirectional, opt.max_path, opt.recurrent_drop, opt.drop)


print(net)
if opt.USE_CUDA == True:
    net.cuda()
    
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0)    


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

train_losses = []
val_losses = []
val_acc_history = []
epoch_train_loss = []

time_per_epoch = 0
num_mini_batch = len(x_train)
    
best_val_score = None
better_model = False
epoch_counter = 0
break_training = 0
break_flag = False

for epoch in range(1, opt.num_epochs + 1):
    net.train()
    epoch_loss = 0
    start_time = time.time()  
    epoch_start = time.time()
    
    if break_flag == True:
        break
    mini_batch_inds = list(range(num_mini_batch))
    for i in mini_batch_inds:
        #x_batch = [max_len  x mini_batch_size x char_encoding_dim]
        x_batch = x_train[i]
        y_batch = y_train[i]
        seg_ind_batch = seg_ind_train[i]
        len_list_batch = batched_len_list_train[i]
        
        bs = x_batch.size(1)
        #Get sorted indices based on length
        sorted_inds_vals = [t for t in sorted(enumerate(len_list_batch), reverse=True, key=lambda x:x[1])]
        sorted_inds, sorted_vals = map(list, zip(*sorted_inds_vals))
        sorted_inds = np.array(sorted_inds)
        sorted_inds = torch.LongTensor(sorted_inds)
        
        if opt.USE_CUDA == True:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            seg_ind_batch = seg_ind_batch.cuda()
            sorted_inds = sorted_inds.cuda()
   
        #Sort sentences of batch based on length
        x_batch_s = torch.index_select(x_batch, 1, sorted_inds )
        y_batch_s = torch.index_select(y_batch, 1, sorted_inds )
        seg_ind_s = torch.index_select(seg_ind_batch, 1, sorted_inds )
        
        #Convert to Variables to get gradient
        x_batch_s = Variable(x_batch_s)
        y_batch_s = Variable(y_batch_s)
        seg_ind_s = Variable(seg_ind_s)
       
        #Pack and unpack so we trim any excessive zeros
        pack_y = torch.nn.utils.rnn.pack_padded_sequence(y_batch_s, sorted_vals)
        unpacked_y = torch.nn.utils.rnn.pad_packed_sequence(pack_y)[0]
        
        pack_seg_ind = torch.nn.utils.rnn.pack_padded_sequence(seg_ind_s, sorted_vals)
        unpacked_seg_ind = torch.nn.utils.rnn.pad_packed_sequence(pack_seg_ind)[0]
        
        optimizer.zero_grad()
        
        hidden = net.init_hidden(bs)
        output_2d, hidden = net( x_batch_s, hidden, sorted_vals )
        
        #Convert this to tensor after you used packing, else error 'torch.LongTensor' object is not reversible
        sorted_vals =  torch.LongTensor(sorted_vals)
        if opt.USE_CUDA == True:
            sorted_vals = sorted_vals.cuda()
        forward_var_batch = net._forward_alg(output_2d, sorted_vals)
        gold_score_batch = net.score(output_2d, unpacked_y, unpacked_seg_ind, sorted_vals)
        
         
        loss = (forward_var_batch-gold_score_batch  ).mean()
        torch.nn.utils.clip_grad_norm(net.parameters(), opt.clip)

        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        

        if i % opt.print_every == 0:
           
            elapsed = time.time() - start_time
            print ('Epoch [%d/%d], Batch [%d/%d], Train Loss: %.4f, s/batch %5.2f' 
                   %(epoch, opt.num_epochs, i+1, num_mini_batch, loss.data[0], elapsed/opt.print_every))
            start_time = time.time() 
            
    epoch_loss /= num_mini_batch    
    epoch_train_loss.append( epoch_loss )
    epoch_end = time.time()
    time_per_epoch += epoch_end - epoch_start  
    
    if  (epoch % opt.epoch_append == 0):
        
        val_score = evaluate(net, x_val, y_val, seg_ind_val, batched_len_list_val, opt)
        
        train_losses.append(loss.data[0])
        val_acc_history.append( val_score )
        print ('Epoch Loss', epoch_loss)
        
        #Save best model
        if not best_val_score or val_score > best_val_score:
            best_val_score = val_score
            with open(model_path, 'wb') as f:
                torch.save(net.state_dict(), f)
            better_model = True
            epoch_counter = 0
            print ('New Best validation accuracy --------- {:.4f}-\n'.format(val_score))
            break_training = 0
        #If we haven't seen a new validation accuracy after opt.epoch_break epochs, stop training  
        elif break_training == opt.epoch_break:
            break_flag = True #Make inner loop a train() function, return and break on outer loop
        #If we haven't seen a new validation accuracy after anneal_lr_every epochs, decrease learning rate    
        elif epoch_counter == opt.anneal_lr_every:
            print ('Learning rate reduced in epoch {:d}'.format(epoch))
            opt.lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
            epoch_counter = 0
        else: 
            break_training += 1
            epoch_counter += 1
            



