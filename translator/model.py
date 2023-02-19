#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
from transformer import Encoder, Decoder, PositionalEncoding
import re
import math

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName, map_location=torch.device('cpu')))
    
    def __init__(self, sourceWord2ind, targetWord2ind, unkToken, padToken, h, d_model, d_ff, n, dropout):
        super(NMTmodel, self).__init__()
        self.source_word2ind, self.target_word2ind = sourceWord2ind, targetWord2ind
        self.source_vocab_len, self.target_vocab_len = len(sourceWord2ind), len(targetWord2ind)

        self.unkTokenIdx = targetWord2ind[unkToken]
        self.padTokenIdx = targetWord2ind[padToken]

        assert d_model % h == 0 
        d_v, d_k = d_model // h, d_model // h

        self.input_embedding = torch.nn.Embedding(self.source_vocab_len, d_model)
        self.output_embedding = torch.nn.Embedding(self.target_vocab_len, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder = Encoder(h, d_model, d_v, d_k, d_ff, n, dropout)
        self.decoder = Decoder(h, d_model, d_v, d_k, d_ff, n, dropout)

        self.fc = torch.nn.Linear(d_model, self.target_vocab_len)

    def forward(self, source, target, enc_mask = None, dec_mask = None, device = 'cuda:0'):

        #x = self.preparePaddedBatch(source, self.source_word2ind)
        x = self.input_embedding(source)
        x_encoded = self.pos_encoding(x)
        encoder_output = self.encoder(x_encoded, enc_mask)

        #y = self.preparePaddedBatch(target, self.target_word2ind)
        y = self.output_embedding(target)
        y_encoded = self.pos_encoding(y)

        # y.shape (batch_size, seq_len, d_model)

        decoder_output = self.decoder(y_encoded, encoder_output, enc_mask, dec_mask)

        output = self.fc(decoder_output)

        return output

def translateSentence(model, sentence, startToken, endToken, beta, limit=1000, device='cuda:0'):
    #sentence = "При всички теми , обсъждани в Съвета , се установи проблем .".split()
    startTokenIdx = model.target_word2ind[startToken]
    endTokenIdx = model.target_word2ind[endToken]

    target_ind2word = {v: k for k, v in model.target_word2ind.items()}

    src = [[model.source_word2ind.get(w, model.unkTokenIdx) for w in sentence]]
    x = torch.tensor(src,dtype=torch.int, device=device)

    trg = [startTokenIdx]
    #y = torch.tensor(trg, dtype=torch.int, device=device)
    candidate_seqs = [(trg, 0)]
    longest = 0
    while longest < limit:
        ys = [torch.tensor(candidate, dtype=torch.int, device=device).unsqueeze(0) for candidate, _ in candidate_seqs if candidate[-1] != endTokenIdx]
        outputs = [model(x, y, device=device) for y in ys]

        outputs = [output.squeeze(0) for output in outputs]
        probs = [torch.nn.functional.softmax(output[-1], dim=0) for output in outputs]
        # best_guess_idx = torch.argmax(probs)
        next_candidates = []
        for p, (cand, s) in zip(probs, candidate_seqs):
            if cand[-1] == endTokenIdx: 
              next_candidates.append((cand,s))
              continue

            probs_n, next_best_n = torch.sort(p, descending=True)
            probs_n, next_best_n = probs_n[:beta], next_best_n[:beta]
            extended_candidate = [(cand + [idx], s + math.log(q)) for idx, q in zip(next_best_n, probs_n)]
            next_candidates.extend(extended_candidate)
        
        candidate_seqs = next_candidates
        sorted(candidate_seqs, key = lambda p: p[1] / len(p[0]), reverse=True)
        candidate_seqs = candidate_seqs[:beta]
        
        if all([candidate[-1] == endTokenIdx for candidate, _ in candidate_seqs]):
            trg = candidate_seqs[0][0][:-1]
            break

        longest += 1
    
    #print(trg[1].item())
    result = [target_ind2word[w_idx.item()] for w_idx in trg[1:]]

    result = ' '.join(result) + "\n"
    result = re.sub(r'[^\S\r\n]([);:?,.!])', r'\1', result)
    result = re.sub(r"([(])[^\S\r\n]", r'\1', result)
    result = re.sub(r"[^\S\r\n](')", r'\1', result)
    ##exit()
    return result