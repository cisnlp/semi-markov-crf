# semi-markov-crf

Code for paper **Neural Semi-Markov Conditional Random Fields for Robust Character-Based Part-of-Speech Tagging**

## Requirements
python 3.6

pytorch 0.2.0

numpy

sklearn


## Train

Go to train.py from arg\_list select the model\_path and the LANG (language) you want to select. The options are:

'en' for English  UD 1.2 

'en20' for English  UD 1.2 

'ja' for Japanese UD 2.0

'zh' for Chinese UD 2.0

'vi' for Vietnamese UD 2.0


You can also select the segment constructor (grConv or SRNN, default grConv)

To train the model with the default parameters now run `python3 train.py`

## Evaluate

After training a model go to predict.py and select the model\_path and the LANG you want to evaluate

Current repo has an untrained model on Vietnamese

## Marmot

For MarMot evaluation (Table 1, 4) run scripts evaluate_marmot_joint.py and evaluate_marmot_relaxed.py


The relaxed evaluation constructs a label for each gold token by taking the label outputs of the UDPipe + Marmot
. 
Four possible tokenization output cases are detected.

1. A correct token was detected: e.g Wonderful.

   The label selected in this case is the predicted

2. A corrupted space tokenization wasn't merged. e.g Wonderful -> Wo-nd-er-ful.

   The label selected in this case is the golden one if any of the sub-tokens had a correct label.  
   For example if any of the (Wo, nd, er, ful) tokens has an ADJ label (Wonderful is adjective), the golden label is selected as output.  
   If none has the golden one, a random wrong label is selected instead.


3. A merge of one or more tokens occured. e.g Don'tgo

   In this case MarMot outputs only 1 label for all three seperate tokens.
   In this case we output the golden label for each of the individual gold token, only if that token has a gold label as the single predicted label, otherwise we output a randomly wrong label for that token

   e.g Don'tgo GOLD -> [(Do) VERB ,(n't) ADV, (go)VERB]   
      predicted -> [ADV],
      constructed output-> [WRONG_LABEL, ADV, WRONG_LABEL]


4. A space tokenization wasn't merged, but also the last sub-token was merged with a next word (or more) (e.g Wonderful world -> Wo-nd-er-fulworld)

...In this case all of the subtokens before the last sub-token (Wo-nd-er) are used to create the label for the first word (Wonderful), in the same way as in case 2.
...The merged case (-fulworld) is treated as in case 3.









