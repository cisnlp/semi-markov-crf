import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time


class GRC(nn.Module):
    def __init__(self, char_enc_size, label_to_ind, rnn_type, emb_size, hidden_size, num_layers,
                  bidirectional,max_path, recurrent_drop=0, input_drop=0):
        super(GRC, self).__init__()
        self.char_embed = nn.Linear(char_enc_size, emb_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size_seg = self.hidden_size
        self.recurrent_drop = recurrent_drop
        
        if self.bidirectional == True:
            self.NUM_DIRECTIONS = 2
        else:
            self.NUM_DIRECTIONS = 1
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_size, self.hidden_size // self.NUM_DIRECTIONS, num_layers, batch_first=False,
                              dropout=self.recurrent_drop, bidirectional = self.bidirectional)
         
        self.WL = nn.Linear(self.hidden_size, self.hidden_size, bias=None)
        self.WR = nn.Linear(self.hidden_size, self.hidden_size, bias=None)
        
        self.bw = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())        
        
        self.GL = nn.Linear(self.hidden_size, 3*self.hidden_size, bias=None)
        self.GR = nn.Linear(self.hidden_size, 3*self.hidden_size, bias=None)
        
        self.bg = nn.Parameter(torch.FloatTensor(1, 3*self.hidden_size).zero_())     
        self.sigm = torch.nn.Sigmoid()    
        
        self.tag_to_ix = label_to_ind
        self.tagset_size = len(self.tag_to_ix)
        self.max_path = max_path
        
        self.drop3d = nn.Dropout3d(input_drop)
        self.drop = nn.Dropout(input_drop)
        
        self.fc = nn.Linear(self.hidden_size_seg,  self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
    
        self.init_weights()


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.NUM_DIRECTIONS, batch_size, self.hidden_size // self.NUM_DIRECTIONS).zero_()),
                        Variable(weight.new(self.num_layers*self.NUM_DIRECTIONS, batch_size, self.hidden_size // self.NUM_DIRECTIONS).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.NUM_DIRECTIONS, batch_size, self.hidden_size // self.NUM_DIRECTIONS).zero_() )    
    
    
    def init_hidden_seg(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(1, batch_size, self.hidden_size_seg //2  ).zero_()),
                        Variable(weight.new(1, batch_size, self.hidden_size_seg //2).zero_()))
        else:
            return Variable(weight.new(1, batch_size, self.hidden_size_seg).zero_() ) 


    def forward(self, x_batch_s, hidden, sorted_vals):
        batch_size = x_batch_s.size(1)
        x_batch_s = self.drop3d(x_batch_s)
        emb = self.char_embed(x_batch_s)
        emb = self.drop(emb)
        
        pack_emb = torch.nn.utils.rnn.pack_padded_sequence(emb, sorted_vals)
        out, hidden = self.rnn(pack_emb, hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)
        
        max_len = unpacked.size(0)
        unpacked = unpacked.transpose(1,0)

        segment_feat = Variable(unpacked.data.new(batch_size, max_len, self.max_path, self.hidden_size).fill_(0))
        segment_feat[:, :, 0, :] = unpacked[:,:,:]
        
        left_feat = unpacked[:,:-1,:].contiguous().view(-1, self.hidden_size )
        right_feat = unpacked[:,1:,:].contiguous().view(-1, self.hidden_size )
        
        for i in range(1, self.max_path):
            next_level_cand = self.WL(left_feat) + self.WR(right_feat) + self.bw
            next_level_cand = 4*(self.sigm(next_level_cand) - 0.5)
            
            gate_feat = self.GL(left_feat) + self.GR(right_feat) + self.bg
            gate_feat = torch.exp(gate_feat)
            
            theta_scores =  tuple( gate_feat[:, i*self.hidden_size:(i+1)*self.hidden_size].unsqueeze(-1) for i in range(3) )
            theta_scores = torch.cat( theta_scores, 2)
            
            Z = torch.sum(theta_scores, 2)
            inv_z = 1/Z.unsqueeze(-1).expand(*Z.size(), 3)
            theta_norm = theta_scores * inv_z
            
            next_level_cand =   torch.mul(left_feat,theta_norm[:,:,0]) + torch.mul(right_feat,theta_norm[:,:,1]) + torch.mul(next_level_cand,theta_norm[:,:,2])
            next_level_cand = next_level_cand.view(batch_size, max_len-i, self.hidden_size)
            
            segment_feat[:, i:, i, :] = next_level_cand
            left_feat = next_level_cand[:,:-1,:].contiguous().view(-1, self.hidden_size )
            right_feat = next_level_cand[:,1:,:].contiguous().view(-1, self.hidden_size )

        #Get tag scores for crf
        segment_feat = self.fc(self.drop(segment_feat.view(-1, self.hidden_size) ))
        segment_feat = segment_feat.view(batch_size, max_len, self.max_path, self.tagset_size)
        
        return segment_feat, hidden    
  
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        for name, param in self.named_parameters(): 
            if ('weight' in name): 
                print ('Initializing ', name) 
                initrange = np.sqrt( 6 / sum(param.size()))
                self.state_dict()[name].uniform_(-initrange, initrange)
             
        
    def _forward_alg(self, logits, len_list, is_volatile=False):
        """
        Computes the (batch_size,) denominator term (FloatTensor list) for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, max_path, n_labels = logits.size()
        
        alpha = logits.data.new(batch_size, seq_len+1, self.tagset_size).fill_(-10000)
        alpha[:, 0, self.tag_to_ix['START']] = 0
        alpha = Variable(alpha, volatile=is_volatile)
        
        # Transpose batch size and time dimensions:
        logits_t = logits.permute(1,0,2,3)
        c_lens = len_list.clone()
        
        alpha_out_sum = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(0))
        mat = Variable(logits.data.new(batch_size,self.tagset_size,self.tagset_size).fill_(0))
        
        for j, logit in enumerate(logits_t):
            for i in range(0,max_path):
                if i<=j:
                    alpha_exp = alpha[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                    logit_exp = logit[:, i].unsqueeze(-1).expand(batch_size, self.tagset_size, self.tagset_size)
                    trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
                    mat = alpha_exp + logit_exp + trans_exp
                    alpha_out_sum[:,i,:] =  log_sum_exp(mat , 2, keepdim=True)
                    
            alpha_nxt = log_sum_exp(alpha_out_sum , dim=1, keepdim=True).squeeze(1)
            
            mask = Variable((c_lens > 0).float().unsqueeze(-1).expand(batch_size,self.tagset_size))
            alpha_nxt = mask * alpha_nxt + (1 - mask) *alpha[:, j, :].clone() 
            
            c_lens = c_lens - 1      

            alpha[:,j+1, :] = alpha_nxt

        alpha[:,-1,:] = alpha[:,-1,:] + self.transitions[self.tag_to_ix['STOP']].unsqueeze(0).expand_as(alpha[:,-1,:])
        norm = log_sum_exp(alpha[:,-1,:], 1).squeeze(-1)

        return norm

        
    def viterbi_decode(self, logits, lens):
        """
        Use viterbi algorithm to compute the most probable path of segments
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, max_path, n_labels = logits.size()
        
        # Transpose to batch size and time dimensions
        logits_t = logits.permute(1,0,2,3)
        
        vit = Variable(logits.data.new(batch_size,seq_len+1, self.tagset_size).fill_(-10000),
                                       volatile = not self.training)
        
        vit_tag_max = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(-10000),
                                   volatile = not self.training) 
        
        vit_tag_argmax = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(-100),
                                   volatile = not self.training) 
        vit[:,0, self.tag_to_ix['START']] = 0
        c_lens = Variable(lens.clone(), volatile= not self.training)
        
        pointers = Variable(logits.data.new(batch_size, seq_len, self.tagset_size, 2 ).fill_(-100))
        for j, logit in enumerate(logits_t):
            for i in range(0,max_path):
                if i<=j:
                    vit_exp = vit[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                    trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
                    vit_trn_sum = vit_exp + trn_exp
                    vt_max, vt_argmax = vit_trn_sum.max(2)
                    vit_nxt = vt_max + logit[:, i]
                    vit_tag_max[:,i,:] = vit_nxt
                    vit_tag_argmax[:,i,:] = vt_argmax
           
            seg_vt_max, seg_vt_argmax = vit_tag_max.max(1)
            
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(seg_vt_max)
            vit[:, j+1, :] = mask*seg_vt_max + (1-mask)*vit[:, j, :].clone()
            
            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(  vit[:, j+1, :])
            vit[:, j+1, :] = vit[:, j+1, :] +  mask * self.transitions[ self.tag_to_ix['STOP'] ].unsqueeze(0).expand_as( vit[:, j+1, :] )
            
            idx_exp = seg_vt_argmax.unsqueeze(1)
            pointers[:,j,:,0] =  torch.gather(vit_tag_argmax, 1,idx_exp ).squeeze(1)
            pointers[:,j,:,1] = seg_vt_argmax 
            
            c_lens = c_lens - 1  
        
        #Get the argmax from the last viterbi scores and follow the reverse pointers for the best path 
        end_max , end_max_idx = vit[:,-1,:].max(1)
        end_max_idx = end_max_idx.data.cpu().numpy()
        
        pointers = pointers.data.long().cpu().numpy()
        pointers_rev = np.flip(pointers,1)
        paths = []
        segments = []
        
        for b in range(batch_size):
            #Different lengths each sentence, so get the starting index on the reverse list
            start_index = seq_len-lens[b] 
            path = [end_max_idx[b]]
            segment = [lens[b]]
            
            if (start_index >= seq_len -1):
                paths.append(path)
                continue
            
            max_tuple = pointers_rev[b,start_index,end_max_idx[b]]
            start_index += 1
            prev_tag = end_max_idx[b]
            next_tag = max_tuple[0]
            next_jump = max_tuple[1]
            
            for j, argmax in enumerate(pointers_rev[b,start_index:,:]):
                #Append same tag as many times as indicated by the best segment length we stored
                if next_jump > 0:
                    next_jump -= 1
                    path.insert(0, prev_tag)
                    continue
                #Switch to next tag when we hit zero
                else:
                    segment.insert(0, lens[b]- j-1)
                    path.insert(0, next_tag)
                
                #Get the next tag, and the number of times we have to append the previous one
                prev_tag = next_tag
                max_tuple = argmax[next_tag]
                next_tag = max_tuple[0]
                next_jump = max_tuple[1]
                
            segments.append(segment)     
            paths.append(path)
            
        return paths, segments
        
        
    def _bilstm_score(self, logits, labels, seg_inds, lens):
        
        """
        Computes the (batch_size,) numerator (FloatTensor list) for the log-likelihood, which is the
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            labels: [batch_size, seq_len] LongTensor
            seg_inds: [batch_size, seq_len] LongTensor
            lens: [batch_size] LongTensor
        """
        lens = Variable( lens, volatile = not self.training)
        
        batch_size, max_len, _, _ = logits.size()
        
        # Transpose to batch size and time dimensions
        labels = labels.transpose(1,0)
        
        seg_inds = seg_inds.transpose(1,0).data.cpu().numpy()
        labels_exp = labels.unsqueeze(-1)

        #Construct the mask the will sellect the corrects segments from all possible segments for each timstep
        mask_seg = np.zeros(( batch_size, max_len, self.max_path))
        
        mask_step =  np.zeros(( batch_size), dtype=np.int32)
        counter = np.zeros((batch_size), dtype=np.int32)
        
        #For each timstep accross all sentences
        for i in range(0,max_len):
            #0 or 1 depending if we are on the end of a segment
            mask_step =  seg_inds[:, i] 
            mask_seg[np.arange(batch_size), i, counter] = mask_step 
            counter = counter + 1
            counter = (1- mask_step)*counter*(counter < self.max_path)
           
        mask_seg = torch.from_numpy(mask_seg).float()
        if next(self.parameters()).is_cuda == True:
            mask_seg = mask_seg.cuda()
            
        mask_seg = mask_seg.unsqueeze(-1).expand_as(logits)
        mask_seg = Variable(mask_seg,  volatile = not self.training) 
        
        logit_mask = logits*mask_seg
        sum_cols = torch.sum(logit_mask, dim=2).squeeze(2)
        
        all_scores = torch.gather(sum_cols, 2, labels_exp).squeeze(-1)
        
        mask_time = sequence_mask(lens).float()
        all_scores = all_scores*mask_time
        
        sum_seg_scores = torch.sum(all_scores, dim=1).squeeze(-1)

        return  sum_seg_scores
        
    def score(self, logits, y, seg_inds, lens):


        bilstm_score = self._bilstm_score(logits, y, seg_inds, lens)
        transition_score = self.transition_score(y, lens, seg_inds )
        
        score = transition_score + bilstm_score

        return score
    
    def transition_score(self, labels, lens, mask_seg_idx):
        """
        Computes the (batch_size,) scores (FloatTensor list) that will be added to the emission scores
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            labels: [batch_size, seq_len] LongTensor
            seg_inds: [batch_size, seq_len] LongTensor
            lens: [batch_size] LongTensor
        """
        lens = Variable( lens, volatile = not self.training)
        labels = labels.transpose(1,0)
        mask_seg_idx = mask_seg_idx.transpose(1,0)
        batch_size, seq_len = labels.size()
        # pad labels with <start> and <stop> indices
        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.tag_to_ix['START']
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.tag_to_ix['STOP']))
        
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 + (-1)*mask) * pad_stop + mask * labels_ext
        trn = self.transitions
        
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels_ext[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)
        
        lbl_lexp = labels_ext[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)
        
        # Mask sentences in time dim
        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        
        trn_scr[:, 1:] = trn_scr[:, 1:].clone()*mask_seg_idx.float() 
        
        score = trn_scr.sum(1).squeeze(-1)
        
        return score

    def loglik(self, logits, y, lens):
        norm_score = self._forward_alg(logits, lens)
        sequence_score = self.score(logits, y, lens, logits=logits)
        loglik = sequence_score - norm_score

        return loglik   


def log_sum_exp(vec, dim=0, keepdim=True):
    max_val, idx = torch.max(vec, dim, keepdim=True)
    max_exp = max_val.expand_as(vec)
    
    return max_val + torch.log(torch.sum(torch.exp(vec - max_exp), dim, keepdim=keepdim))

    
def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)
    if max_len is None:
        
        max_len = lens.max().data[0]
            
    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)
    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask