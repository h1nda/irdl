import sys
import torch
import train
import model
import pickle
from parameters import device, corpusFileName, modelFileName, trainDataFileName, testDataFileName, char2idFileName, auth2idFileName
import optuna

startChar = '{'
endChar = '}'
unkChar = '@'
padChar = '|'

testCorpus = pickle.load(open(testDataFileName, 'rb'))
trainCorpus = pickle.load(open(trainDataFileName, 'rb'))
char2id = pickle.load(open(char2idFileName, 'rb'))
auth2id = pickle.load(open(auth2idFileName, 'rb'))

def objective(trial):
    embed_size = trial.suggest_int("embed_size", 16, 64)
    hidden_size = trial.suggest_int("hidden_size",24, 256)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 10) 
    dropout = trial.suggest_float("dropout", 0.4, 1)
    lr = trial.suggest_float("lr", 0.001, 1)
    epochs = trial.suggest_int("epochs", 5, 100)
    batchSize = trial.suggest("batchSize", 2, 32)

    lm = model.LSTMLanguageModelPack(embed_size, hidden_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(lm.parameters(), lr=lr)
    train.trainModel(trainCorpus, lm, optimizer, epochs, batchSize)

    return train.perplexity(lm, testCorpus, batchSize)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)