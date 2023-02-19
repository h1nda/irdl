#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *

startToken = '<S>'
endToken = '</S>'
unkToken = '<UNK>'
padToken = '<PAD>'

def perplexity(nmt, sourceTest, targetTest, batchSize):
    testSize = len(sourceTest)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        sourceBatch = sourceTest[b:min(b+batchSize, testSize)]
        targetBatch = targetTest[b:min(b+batchSize, testSize)]
        l = sum(len(s)-1 for s in targetBatch)
        c += l
        with torch.no_grad():
            src, trg = nmt.preparePaddedBatch(sourceBatch, sourceWord2ind), nmt.preparePaddedBatch(targetBatch, targetWord2ind)
            
            trg_input = trg[:,:-1]
            trg_target = trg[:,1:].flatten(0,1)

            batch_size, seq_len = trg_input.shape

            dec_peek_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).expand(batch_size, 1, seq_len, seq_len)
            # nuli sa koito ne trqbva da vijda
            # dec_peek_mask.shape = (batch_size, 1, seq_len, seq_len)

            enc_pad_mask = (src != nmt.padTokenIdx).unsqueeze(1).unsqueeze(1).to(device)
            # nuli sa koito sa padding index
            # enc_pad_mask.shape = (batch_size, 1, 1, seq_len) 
            dec_pad_mask = (trg_input != nmt.padTokenIdx).unsqueeze(1).unsqueeze(1).to(device)

            dec_mask = dec_peek_mask & dec_pad_mask

            output = nmt(src, trg_input, enc_pad_mask, dec_mask)
            output = output.flatten(0,1)

            loss = torch.nn.functional.cross_entropy(input=output, target=trg_target, ignore_index=nmt.padTokenIdx)
            H += l * loss
    return math.exp(H/c)

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    sourceCorpus, sourceWord2ind, targetCorpus, targetWord2ind, sourceDev, targetDev = utils.prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken)
    pickle.dump((sourceCorpus,targetCorpus,sourceDev,targetDev), open(corpusDataFileName, 'wb'))
    pickle.dump((sourceWord2ind,targetWord2ind), open(wordsDataFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    (sourceCorpus,targetCorpus,sourceDev,targetDev) = pickle.load(open(corpusDataFileName, 'rb'))
    (sourceWord2ind,targetWord2ind) = pickle.load(open(wordsDataFileName, 'rb'))

    nmt = model.NMTmodel(sourceWord2ind, targetWord2ind, unkToken, padToken, h, d_model, d_ff, n, dropout).to(device)
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (bestPerplexity,learning_rate,osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf

    idx = np.arange(len(sourceCorpus), dtype='int32')
    nmt.train()
    trial = 0
    patience = 0
    iter = 0
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        targetWords = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
            iter += 1
            sourceBatch = [ sourceCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            targetBatch = [ targetCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            
            st = sorted(list(zip(sourceBatch,targetBatch)),key=lambda e: len(e[0]), reverse=True)
            (sourceBatch,targetBatch) = tuple(zip(*st))
            targetWords += sum( len(s)-1 for s in targetBatch )

            src, trg = nmt.preparePaddedBatch(sourceBatch, sourceWord2ind), nmt.preparePaddedBatch(targetBatch, targetWord2ind)

            trg_input = trg[:,:-1]
            trg_target = trg[:,1:].flatten(0,1)

            batch_size, seq_len = trg_input.shape

            dec_peek_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).expand(batch_size, 1, seq_len, seq_len)
            # nuli sa koito ne trqbva da vijda
            # dec_peek_mask.shape = (batch_size, 1, seq_len, seq_len)

            enc_pad_mask = (src != nmt.padTokenIdx).unsqueeze(1).unsqueeze(1).to(device)
            # nuli sa koito sa padding index
            # enc_pad_mask.shape = (batch_size, 1, 1, seq_len) 
            dec_pad_mask = (trg_input != nmt.padTokenIdx).unsqueeze(1).unsqueeze(1).to(device)

            dec_mask = dec_peek_mask & dec_pad_mask

            output = nmt(src, trg_input, enc_pad_mask, dec_mask)
            output = output.flatten(0,1)

            H = torch.nn.functional.cross_entropy(input=output, target=trg_target, ignore_index=nmt.padTokenIdx, label_smoothing = eps_ls)

            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, ", loss: ",H.item(), "words/sec:",targetWords / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                targetWords = 0
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, sourceDev, targetDev, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    patience = 0
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')
                else:
                    patience += 1
                    if patience == max_patience:
                    
                        trial += 1
                        if trial == max_trials:
                            print('early stop!')
                            exit(0)
                        learning_rate *= learning_rate_decay
                        print('load previously best model and decay learning rate to:', learning_rate)
                        nmt.load(modelFileName)
                        (bestPerplexity,_,osd) = torch.load(modelFileName + '.optim')
                        optimizer.load_state_dict(osd)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                        patience = 0

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, sourceDev, targetDev, batchSize)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    (sourceWord2ind,targetWord2ind) = pickle.load(open(wordsDataFileName, 'rb'))
    
    nmt = model.NMTmodel(sourceWord2ind, targetWord2ind, unkToken, padToken, h, d_model, d_ff, n, dropout).to(device)
    nmt.load(modelFileName)
    
    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    targetTest = [ [startToken] + s + [endToken] for s in targetTest]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, sourceTest, targetTest, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    (sourceWord2ind,targetWord2ind) = pickle.load(open(wordsDataFileName, 'rb'))

    sourceTest = utils.readCorpus(sys.argv[2])

    nmt = model.NMTmodel(sourceWord2ind, targetWord2ind, unkToken, padToken, h, d_model, d_ff, n, dropout).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3],'w')
    pb = utils.progressBar()
    pb.start(len(sourceTest))
    for s in sourceTest:
        file.write(model.translateSentence(nmt, s, startToken, endToken, beta))
        pb.tick()
    pb.stop()

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
