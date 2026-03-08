import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n_items, dim=64, n_heads=2, n_layers=2, max_len=50):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def encode(self, seq):
        b, l = seq.shape
        pos = torch.arange(l, device=seq.device).unsqueeze(0).expand(b, l)
        x = self.item(seq) + self.pos(pos)
        return self.enc(x)[:, -1, :]

    def score_items(self, seq, item_ids):
        h = self.encode(seq)
        e = self.item(item_ids)
        return (h * e).sum(-1)
