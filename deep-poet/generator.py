#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch

def generateText(model, char2id, auth, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:] if len(startSentence)>1 else startSentence

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.
    # source: [(author, list of characters)]
    author_idx = model.auth2id.get(auth, 0)
    auth_embedding = model.author_embed(torch.tensor([author_idx], device="cuda:0"))

    h_0 = auth_embedding.repeat(model.num_layers, 1)
    c_0 = auth_embedding.repeat(model.num_layers, 1)

    for i in range(limit):
        sent_tensor = torch.t(torch.tensor([char2id.get(c, model.unkTokenIdx) for c in result[i:]], device="cuda:0"))
        sent_embedding = model.embed(sent_tensor)
        
        _, (h_n, c_n) = model.lstm(sent_embedding, (h_0, c_0))
        
        h_n_last = h_n[-1,:]
        # print(h_n_last)
        drop = model.dropout(h_n_last)
        dense = model.fc(drop)

        probs = torch.nn.functional.softmax(dense/temperature, dim=0)
        p = probs.numpy(force=True)
        p = p / np.sum(p)

        next_symbol = np.random.choice(list(char2id.keys()), p=p)

        if model.word2ind[next_symbol] == model.endTokenIdx:
            break

        h_0, c_0 = h_n, c_n

        result += next_symbol
    #### Край на Вашия код
    #############################################################################

    return result if result[0] != '{' else result[1:]
