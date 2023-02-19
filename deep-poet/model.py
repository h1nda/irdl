#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import torch

#################################################################
####  LSTM с пакетиране на партида
#################################################################

class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a,s) in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for (a,s) in source]
        auths = [self.auth2id.get(a,0) for (a,s) in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device)), torch.tensor(auths, dtype=torch.long, device=device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName,device))

    def __init__(self, embed_size, hidden_size, auth2id, word2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН, влагане за автора и dropout
        #############################################################################
        #### Начало на Вашия код.
        self.word2ind = word2ind
        self.auth2id = auth2id

        self.vocab_size = len(word2ind)
        self.author_size = len(auth2id)

        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]

        self.embed = torch.nn.Embedding(self.vocab_size, embed_size)
        self.author_embed = torch.nn.Embedding(self.author_size, hidden_size)

        self.num_layers = lstm_layers

        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=self.num_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, self.vocab_size)
        #### Край на Вашия кодs
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout и началните скрити вектори
        #############################################################################
        #### Начало на Вашия код.
        # Една поема е една колона в X!!!!
        # А е вектор от индексите на авторите в бача 
        X, A = self.preparePaddedBatch(source)
        E = self.embed(X[:-1])
        author_embedding = self.author_embed(A)

        h_0 = author_embedding.repeat(self.num_layers, 1, 1)
        c_0 = author_embedding.repeat(self.num_layers, 1, 1)

        sequence_lengths = [len(s[1])-1 for s in source]
        output_packed, (h_n, c_n) = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, sequence_lengths, enforce_sorted=False), (h_0,c_0))

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed)

        drop = self.dropout(output.flatten(0,1))
        dense = self.fc(drop)

        Y = X[1:].flatten(0,1)

        x_entropy = torch.nn.functional.cross_entropy(dense, Y, ignore_index=self.padTokenIdx)
        return x_entropy
        #### Край на Вашия код
        #############################################################################

