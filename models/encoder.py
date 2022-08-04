import math
import torch
import torch.nn as nn
from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class GRUEncoder(nn.Module):
    def __init__(self, d_model, dropout, num_inter_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.gru = nn.GRU(d_model, d_model, num_inter_layers, dropout=dropout, bidirectional=True,
                          batch_first=True)
        self.linear = nn.Linear(d_model * 4, d_model * 2, bias=True)
        self.final = nn.Linear(d_model * 2, 200, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        x = top_vecs * mask[:, :, None].float()
        x = self.layer_norm(x)
        output, hidden = self.gru(x)

        x = self.linear(torch.cat([output[0][-1].unsqueeze(0), torch.cat([hidden[0], hidden[1]], 1)], 1))
        x = self.final(x)
        sent_scores = x[:, :mask.shape[1]].squeeze(-1) * mask.float()
        # sent_scores = x[:mask.shape[1]].squeeze(-1) * mask.float()

        return sent_scores
