import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")


beta = 1
h = 4
d_model = 512
d_ff = 512
n = 5
dropout = .2
eps_ls = .2

learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 64

maxEpochs = 20
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
