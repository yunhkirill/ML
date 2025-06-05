import torch
import torch.nn as nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding

from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, source_vocab_size, target_vocab_size, d_model=300, d_ff=1024,
                 blocks_count=4, heads_count=6, dropout_rate=0.1):
        super(EncoderDecoder, self).__init__()

        self.d_model = d_model

        navec = Navec.load("data/navec_hudlit_v1_12B_500K_300d_100q.tar")
        self._emb = nn.Sequential(
            NavecEmbedding(navec),
            PositionalEncoding(d_model, dropout_rate)
        )
        
        for param in self._emb[0].parameters():
            param.requires_grad = False

        self.encoder = Encoder(self._emb, source_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)
        self.decoder = Decoder(self._emb, target_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)

        for p in self.parameters():
            if p.dim() > 1 and p.dtype in [torch.float32, torch.float64]:
                nn.init.xavier_uniform_(p)

    def forward(self, source_inputs, target_inputs, source_mask, target_mask):
        encoder_output = self.encoder(source_inputs, source_mask)
        return self.decoder(target_inputs, encoder_output, source_mask, target_mask)