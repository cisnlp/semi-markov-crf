import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time




class SRNN(nn.Module):
    def __init__(self, char_enc_size, label_to_ind, rnn_type, emb_size, hidden_size, num_layers,
                  bidirectional,max_path, recurrent_drop=0, input_drop=0):
        super(SRNN, self).__init__()
        
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
            self.rnn_seg_fwd = getattr(nn, rnn_type)(self.hidden_size, self.hidden_size_seg //2, 1, batch_first=False,
                              dropout=0, bidirectional = False)
            self.rnn_seg_rev = getattr(nn, rnn_type)(self.hidden_size, self.hidden_size_seg //2, 1, batch_first=False,
                              dropout=0, bidirectional = False)
            
        self.tag_to_ix = label_to_ind
        self.tagset_size = len(self.tag_to_ix)
        self.max_path = max_path
        
        self.drop = nn.Dropout(input_drop)
        self.drop3d = nn.Dropout3d(input_drop)
        self.fc = nn.Linear(self.hidden_size_seg,  self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
    
        self.transitions.data[self.tag_to_ix['START'], :] = -10000
        self.transitions.data[:, self.tag_to_ix['STOP']] = -10000
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

        segment_feat = Variable(unpacked.data.new(batch_size, max_len, self.max_path, self.hidden_size_seg).fill_(0))

        segment_feat_fwd = Variable(unpacked.data.new(batch_size, max_len, self.max_path, self.hidden_size_seg//2).fill_(0))
        hidden_fwd = unpacked[:,:, :]
        #For every sentence in batch
        for b, rnn_features in enumerate(hidden_fwd):
              seg_bs = sorted_vals[b]
              
              #Initialize with zero hidden state for segments of length 1
              hidden_seg = self.init_hidden_seg(seg_bs)
              #input dim [time=1 x bs x hidd_size]
              seg_rnn_input_orig = rnn_features[:sorted_vals[b]].unsqueeze(0)  
              seg_rnn_input = seg_rnn_input_orig.clone() 
              for i in range(0, min(self.max_path, seg_bs)):
                  #Get [1x seg_bs-i x hidd_size] hidden states for segment length i
                  out , seg_encoding = self.rnn_seg_fwd(seg_rnn_input, hidden_seg)
                  #Store segment of length i
                  segment_feat_fwd[b, i:seg_bs, i, :] = out.squeeze(0) 
                  #End of sequence, dont get the next hidden it will raise a null sequence error
                  if (i == seg_bs-1):
                      continue
                  hidden_seg = ( seg_encoding[0][:, :-1, :] , seg_encoding[1][:, :-1, :])
                  #shift starting input sequence to the right by one
                  seg_rnn_input = seg_rnn_input_orig[:, i+1:].clone()
          #    print ('SEG_RNN INPUT', seg_rnn_input.size(), hidden_seg[0].size())
              
        if self.bidirectional == True:
          segment_feat_rev = Variable(unpacked.data.new(batch_size, max_len, self.max_path, self.hidden_size_seg//2).fill_(0))
          hidden_rev = unpacked[:,:, :]
          for b, rnn_features in enumerate(hidden_rev):
                  seg_bs = sorted_vals[b]
                  #Initialize with zero hidden state for segments of length 1
                  hidden_seg = self.init_hidden_seg(seg_bs)
                  #input dim [time=1 x bs x hidd_size]
                  seg_rnn_input_orig = rnn_features[:sorted_vals[b]].unsqueeze(0)  
                  seg_rnn_input = seg_rnn_input_orig.clone() 
                  for i in range(0, min(self.max_path, seg_bs)):
                      #Get [1x seg_bs-i x hidd_size] hidden states for segment length i
                      out , seg_encoding = self.rnn_seg_rev(seg_rnn_input, hidden_seg)
                      segment_feat_rev[b, i:seg_bs, i, :] = out.squeeze(0) 
                      
                      if (i == seg_bs-1):
                          continue
                      hidden_seg = ( seg_encoding[0][:, 1:, :] , seg_encoding[1][:, 1:, :])
                      #input shifts to the left by one each time
                      seg_rnn_input = seg_rnn_input_orig[:, :seg_bs-i-1].clone()
          
          #Concatenate the forward and backward segment encoding
          segment_feat = torch.cat( (segment_feat_fwd, segment_feat_rev), dim=3 ) 
        else:
          segment_feat = segment_feat_fwd

        #Get tag scores for crf
        segment_feat = self.fc( segment_feat.view(-1, self.hidden_size_seg ) )
        segment_feat = segment_feat.view(batch_size, max_len, self.max_path, self.tagset_size)
        
        return segment_feat, hidden    
  
    def init_weights(self):
        self.fc.bias.data.fill_(0)
     #   weight_init.xavier_uniform(self.fc.weight.data, gain=nn.init.calculate_gain('tanh'))
        for name, param in self.named_parameters(): 
            if ('weight' in name): #initiale with [- 1/sqrt(H) ,- 1/sqrt(H)]
                print ('Initializing ', name) 
# =============================================================================
#                 if (name == 'char_embed.weight'):
#                      initrange = np.sqrt( 3 / sum(param.size(0))) #Hovy init sqrt(3/dim)
#                      self.state_dict()[name].uniform_(-initrange, initrange)
#                      continue
# =============================================================================
                initrange = np.sqrt( 6 / sum(param.size()))
                self.state_dict()[name].uniform_(-initrange, initrange)
             #   print (self.state_dict()[name])
             
        
    def _forward_alg(self, logits, len_list, is_volatile=False):
        """
        Computes the (batch_size,) denominator term (FloatTensor list) for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        if self.training == False:
            is_volatile = True
        
        batch_size, seq_len, max_path, n_labels = logits.size()
        
        #Every tag has a zero exp score at start, +1 for the init alphas
        alpha = logits.data.new(batch_size, seq_len+1, self.tagset_size).fill_(-10000)
        
        #Except the START tag
        alpha[:, 0, self.tag_to_ix['START']] = 0
        alpha = Variable(alpha, volatile=is_volatile)
        
        # Transpose batch size and time dimensions:
        logits_t = logits.permute(1,0,2,3)
        c_lens = len_list.clone()
        
        #Variable that will hold all the alpha scores to be summed for each segment i at timestep t
        alpha_out_sum = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(0))
        #Temp variable that sums the scores for segment i at timestep t
        mat = Variable(logits.data.new(batch_size,self.tagset_size,self.tagset_size).fill_(0))
        
        #For each timestep j
        for j, logit in enumerate(logits_t):
            #Consider up to max_path possible segments
            for i in range(0,max_path):
                if i<=j:
                    #Get alpha scores of segment length i+1 and copy them across rows in order to add them for each next tag
                    alpha_exp = alpha[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                    
                    #Get scores for segment i and copy them for each next tag
                    logit_exp = logit[:, i].unsqueeze(-1).expand(batch_size, self.tagset_size, self.tagset_size)
                    
                    #Expand transition matrix for each batch sample
                    trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)

                    mat = alpha_exp + logit_exp + trans_exp
                    
                    #Sum column wise to get the temp alpha score for segment i
                    alpha_out_sum[:,i,:] =  log_sum_exp(mat , 2, keepdim=True)
                    
            #Sum for each label the alpha scores for each segment length for timestep j        
            alpha_nxt = log_sum_exp(alpha_out_sum , dim=1, keepdim=True).squeeze(1)
            
            #Get a mask that decides what batches should get updated or hold the previous value if they exceeded their length
            mask = Variable((c_lens > 0).float().unsqueeze(-1).expand(batch_size,self.tagset_size))
            alpha_nxt = mask * alpha_nxt + (1 - mask) *alpha[:, j, :].clone() 
            
            #Decrease maximum length as we consume an input
            c_lens = c_lens - 1      

            #Store the alpha for timestep j                  
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
        decode_time = time.time()
        batch_size, seq_len, max_path, n_labels = logits.size()
        
        # Transpose to batch size and time dimensions
        logits_t = logits.permute(1,0,2,3)
        
        # Start with everything totally unlikely, the zeroth element holds the initial viterbi variables
        vit = Variable(logits.data.new(batch_size,seq_len+1, self.tagset_size).fill_(-10000),
                                       volatile = not self.training)
        
        #Holds the maximum value for each segment i for each tag at timestep j
        vit_tag_max = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(-10000),
                                   volatile = not self.training) 
        
        #Holds the correspending tag that vit_tag_max was selected from
        vit_tag_argmax = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(-100),
                                   volatile = not self.training) 
        vit[:,0, self.tag_to_ix['START']] = 0
        c_lens = Variable(lens.clone(), volatile= not self.training)
        
        #First index last column holds the argmax indices for each tag for each timestep j, the second index holds the segment length this score came
        pointers = Variable(logits.data.new(batch_size, seq_len, self.tagset_size, 2 ).fill_(-100))
        for j, logit in enumerate(logits_t):
            for i in range(0,max_path):
                if i<=j:
                    #Get the viterbi variables of segmet length i+1 of the previous timestep, and copy them across rows in order to add them for each next tag
                    vit_exp = vit[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                    
                    #Expand transition matrix for each batch sample
                    trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
                    
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    vit_trn_sum = vit_exp + trn_exp
                    
                    #Get the maximum value and maximum tag that has it for segment length i
                    vt_max, vt_argmax = vit_trn_sum.max(2)
                    
                    #Add the emission scores
                    vit_nxt = vt_max + logit[:, i]
                    
                    #Store the args and max for each segment length i
                    vit_tag_max[:,i,:] = vit_nxt
                    vit_tag_argmax[:,i,:] = vt_argmax
           
            #For each tag, get the maximum score and timestep that caused it, this will be the next viterbi variable
            seg_vt_max, seg_vt_argmax = vit_tag_max.max(1)
            
            #Assign the viterbi variable for timestep j or copy the previous one if sentence length has been exceeded
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(seg_vt_max)
            vit[:, j+1, :] = mask*seg_vt_max + (1-mask)*vit[:, j, :].clone()
            
            #If we are on the last timestep of the sentence add the transition from every tag to the STOP state (STOP row of the transition matrix) 
            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(  vit[:, j+1, :])
            vit[:, j+1, :] = vit[:, j+1, :] +  mask * self.transitions[ self.tag_to_ix['STOP'] ].unsqueeze(0).expand_as( vit[:, j+1, :] )
            
            #Store best segment argmax tag for each tag for timestep j (argmax of argmax)  
            idx_exp = seg_vt_argmax.unsqueeze(1)
            pointers[:,j,:,0] =  torch.gather(vit_tag_argmax, 1,idx_exp ).squeeze(1)
            
            #Store the segment length i for each argmax above
            pointers[:,j,:,1] = seg_vt_argmax #j + (-1)*
            
            #We consumed one input, reduce counter by 1
            c_lens = c_lens - 1  
        
        #Get the argmax from the last viterbi scores and follow the reverse pointers for the best path 
        end_max , end_max_idx = vit[:,-1,:].max(1)
        end_max_idx = end_max_idx.data.cpu().numpy()
        
        pointers = pointers.data.long().cpu().numpy()
        pointers_rev = np.flip(pointers,1)
        paths = []
        segments = []
        #For each sentence in batch
        for b in range(batch_size):
            
            #Different lengths each sentence, so get the starting index on the reverse list
            start_index = seq_len-lens[b] 
            path = [end_max_idx[b]]
            segment = [lens[b]]
            
            #Sentence with only 1 letter
            if (start_index >= seq_len -1):
                paths.append(path)
                continue
            #Get next tag and segment length it came from
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
                
            paths.append(path)
            segments.append(segment)   
            
        end_time = time.time()
        print( 'Decode time' , end_time - decode_time)
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
        
        #Same for seg_inds
        seg_inds = seg_inds.transpose(1,0).data.cpu().numpy()
        labels_exp = labels.unsqueeze(-1)

        #Construct the mask the will sellect the corrects segments from all possible segments for each timstep
        mask_seg = np.zeros(( batch_size, max_len, self.max_path))
        
        #This temp variable will hold the seg_ind flag across batches in each timestep
        mask_step =  np.zeros(( batch_size), dtype=np.int32)
        #Counter will be 1 once a segment has been traversed
        counter = np.zeros((batch_size), dtype=np.int32)
        
        #For each timstep accross all sentences
        for i in range(0,max_len):
            #0 or 1 depending if we are on the end of a segment
            mask_step =  seg_inds[:, i] 
            mask_seg[np.arange(batch_size), i, counter] = mask_step 
            counter = counter + 1
            #zero counter if we selected a segment or we exceeded the maximum segment length
            counter = (1- mask_step)*counter*(counter < self.max_path)
           
        mask_seg = torch.from_numpy(mask_seg).float()
        if next(self.parameters()).is_cuda == True:
            mask_seg = mask_seg.cuda()
            
        mask_seg = mask_seg.unsqueeze(-1).expand_as(logits)
        mask_seg = Variable(mask_seg,  volatile = not self.training) 
        
        #Select the correct segment score vectors
        logit_mask = logits*mask_seg
        
        #Sum on the max_path dimension to get rid of the zero segments for each timstep and keep only the score vectors
        sum_cols = torch.sum(logit_mask, dim=2).squeeze(2)
        
        #From the correct segment vectors, select the appropirate tag dimension
        all_scores = torch.gather(sum_cols, 2, labels_exp).squeeze(-1)
        
        #Sentence length mask across batches, zeros excessive scores
        mask_time = sequence_mask(lens).float()
        all_scores = all_scores*mask_time
        
        #Sum on the time dimension to get the final score for each batch
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
        
        # obtain rows from the transition matrix  ,we will transition to in batch and timestep
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels_ext[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)
        
        # obtain intersections of columns from the transition matrix we will transition from and the rows
        # we will transition to, in batch and timestep
        lbl_lexp = labels_ext[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)
        
        # Mask sentences in time dim
        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        
        #1 means get transition for j to j+1, keep all the transitions from start
        trn_scr[:, 1:] = trn_scr[:, 1:].clone()*mask_seg_idx.float() 
        
        #Sum on time dimension to get the transition scores for each sentence
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