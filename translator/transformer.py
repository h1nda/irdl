import torch

class PositionalEncoding(torch.nn.Module):

	def __init__(self, d_model, max_len = 5000):
		super().__init__()

		pe = torch.zeros(1, max_len, d_model)

		position = torch.arange(max_len).unsqueeze(0).unsqueeze(2)
		div_term = ( 10000.0 ** (torch.arange(0, d_model, 2)/d_model) ).unsqueeze(0).unsqueeze(0)
		pe[0,:,0::2] = torch.sin(position / div_term)
		pe[0,:,1::2] = torch.cos(position / div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:,:x.shape[1],:]		# x.shape = (batch_size, seq_len, embedding_dim)
		return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d_model, d_v, d_k):
        super().__init__()
        self.h = h
        self.d_model = d_model
        
        self.d_v = d_v
        self.d_k = d_k

        self.scaling_factor = 1 / (d_k ** .5)

        self.W_Q = torch.nn.Linear(in_features=d_model, out_features=h*d_k, bias = False)
        self.W_K = torch.nn.Linear(in_features=d_model, out_features=h*d_k, bias = False)
        self.W_V = torch.nn.Linear(in_features=d_model, out_features=h*d_v, bias = False)
        self.W_O = torch.nn.Linear(in_features=h*d_v, out_features=d_model, bias = False)
    
    def forward(self, queries, keys, values, mask):
        # keys.shape, values.shape = (batch_size, seq_len, d_model)
        # queries.shape = (batch_size, seq_len_a, d_model)
        batch_size, seq_len, _ = keys.shape
        seq_len_a = queries.shape[1]
        K, V, Q = self.W_K(keys), self.W_V(values), self.W_Q(queries)

        # K.shape, Q.shape = (batch_size, seq_len, d_model)
        # V.shape = (batch_size, seq_len_a, d_model)

        # За scaled dot-product attention трябва да се изчисли QK^T:
        # тоест трябва Q.shape = (batch_size, h, seq_len_a, d_k) и K_t.shape = (batch_size, h, d_k, seq_len)
        Q_reshaped = Q.reshape(batch_size, seq_len_a, self.h, self.d_k).transpose(1,2)
        K_reshaped_t = K.reshape(batch_size, seq_len, self.h, self.d_k).permute(0, 2, 3, 1)
        dot_product = torch.matmul(Q_reshaped, K_reshaped_t)
        # dot_product.shape = (batch_size, h, seq_len_a, seq_len)

        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, -float('inf'))
        
        soft_max = torch.nn.functional.softmax(dot_product * self.scaling_factor, dim=3)
        # soft_max.shape = (batch_size, h, seq_len_a, seq_len)

        # Трябва V.shape = (batch_size, h, seq_len, d_v)
        V_reshaped = V.reshape(batch_size, seq_len, self.h, self.d_v).transpose(1, 2)

        attention = torch.matmul(soft_max, V_reshaped)
        # attention.shape = (batch_size, h, seq_len_a, d_v)

        # За прилагане на W_O трябва attention.shape = (batch_size, seq_len_as, h*d_v)
        output = attention.transpose(1, 2).flatten(start_dim = 2, end_dim = 3)
        
        return self.W_O(output) # (batch_size, seq_len_a, d_model)

class TransformerCell(torch.nn.Module):
    def __init__(self, h, d_model, d_v, d_k, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(h, d_model, d_v, d_k)

        self.ln1 = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(p=dropout)

        #FFN(x) = max(0,xW1 +b1)W2 +b2
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff), # W1,b1
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model) # W2, b2
        )

        self.ln2 = torch.nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, mask = None):
        mha_out = self.mha(queries, keys, values, mask)
        # mha_out.shape = (batch_size, seq_len, d_model)

        ln1_out = self.ln1(self.dropout(mha_out) + queries)

        ffn_out = self.ffn(ln1_out)

        ln2_out = self.ln2(self.dropout(ffn_out) + ln1_out)
        
        return ln2_out

class Encoder(torch.nn.Module):
    def __init__(self, h, d_model, d_v, d_k, d_ff, n, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.cells = torch.nn.ModuleList([TransformerCell(h, d_model, d_v, d_k, d_ff, dropout) for _ in range(n)])

    def forward(self, x, mask = None):
        # x.shape = (batch_size, seq_len, )
        inputs = self.dropout(x)

        for cell in self.cells:
            inputs = cell(inputs, inputs, inputs, mask)
        
        return inputs
    
class DecoderCell(torch.nn.Module):
    def __init__(self, h, d_model, d_v, d_k, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(h, d_model, d_v, d_k)
        self.ln = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(p = dropout)
        self.cell =  TransformerCell(h, d_model, d_v, d_k, d_ff, dropout)
    
    def forward(self, x, keys, values, enc_mask, dec_mask):
        mha_out = self.mha(x, x, x, dec_mask)
        queries = self.ln(self.dropout(mha_out) + x)
        out = self.cell(queries, keys, values, enc_mask)
        return out
        
class Decoder(torch.nn.Module):
    def __init__(self, h, d_model, d_v, d_k, d_ff, n, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(p = dropout)

        self.cells = torch.nn.ModuleList([DecoderCell(h, d_model, d_v, d_k, d_ff, dropout) for _ in range(n)])

    def forward(self, x, encoder_output, enc_mask = None, dec_mask = None):
        # x.shape = (batch_size, seq_len)
        inputs = self.dropout(x)

        for decoder_cell in self.cells:
            inputs = decoder_cell(inputs, encoder_output, encoder_output, enc_mask, dec_mask)
        
        return inputs