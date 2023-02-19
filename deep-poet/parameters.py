import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'
auth2idFileName = 'auth2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 32
char_emb_size = 32

hid_size = 128
lstm_layers = 3
dropout = 0.7

epochs = 50
learning_rate = 0.001

defaultTemperature = 0.9
