from .layers import *
from .mmdit.mmdit_generalized_pytorch import MMDiT

class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out

class DoubleTransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block_1 = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ca_block_2 = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block_1(h1, music, emb, key_padding_mask)
        h2 = h2 + h1
        h3 = self.ca_block_2(h1, y, emb, key_padding_mask)
        h3 = h3 + h2
        out = self.ffn(h3, emb)
        out = out + h3
        return out


class MMDiTBlock(nn.Module):
    def __init__(self, latent_dim=512, num_heads=8, ff_size=1024, dropout=0.1, **kwargs):
        super().__init__()
        self.mmdit = MMDiT(
            depth=1,
            dim_modalities=(latent_dim, latent_dim, latent_dim),
            dim_cond=latent_dim,
            qk_rmsnorm=True
        )

    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        modality_tokens = (x, y, music)
        modality_masks = (key_padding_mask, key_padding_mask, key_padding_mask)

        x_out, y_out, music_out = self.mmdit(
            modality_tokens=modality_tokens,
            modality_masks=modality_masks,
            time_cond=emb
        )

        # Return only the updated x_out for compatibility with previous design
        return x_out
