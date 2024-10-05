import math

import torch
from torch import nn

from utils.embedders import TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TMSA(nn.Module):
    """
    Time-aware Multi-Head Self-Attention (TMSA) module.
    This module combines spatial and temporal embeddings into a unified attention mechanism.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize the TMSA module.

        :param d_model: Dimensionality of the input embeddings.
        :param num_heads: Number of attention heads. d_model must be divisible by num_heads.
        """
        super(TMSA, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for attention across spatial and temporal embeddings
        self.W_qs = nn.Linear(d_model, d_model)
        self.W_qt = nn.Linear(d_model, d_model)
        self.W_ks = nn.Linear(d_model, d_model)
        self.W_kt = nn.Linear(d_model, d_model)
        self.W_vs = nn.Linear(d_model, d_model)
        self.W_vt = nn.Linear(d_model, d_model)

        # Attention bias for controlling the attention weights
        self.attn_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Perform scaled dot-product attention over the query (Q), key (K), and value (V) tensors.

        :param Q: Query tensor of shape (batch_size, num_heads, seq_length, d_k).
        :param K: Key tensor of shape (batch_size, num_heads, seq_length, d_k).
        :param V: Value tensor of shape (batch_size, num_heads, seq_length, d_k).
        :param mask: Optional mask tensor for preventing attention to certain positions.
        :return: Output tensor of shape (batch_size, num_heads, seq_length, d_k).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_scores += self.attn_bias  # Add bias to the attention scores
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # Mask invalid positions
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        Split the input tensor into multiple attention heads.

        :param x: Input tensor of shape (batch_size, seq_length, d_model).
        :return: Tensor of shape (batch_size, num_heads, seq_length, d_k).
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine the multiple attention heads back into a single output.

        :param x: Input tensor of shape (batch_size, num_heads, seq_length, d_k).
        :return: Tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Xs, Xt, mask=None):
        """
        Forward pass of the TMSA module.

        :param Xs: A Tensor of shape (batch_size, seq_len, d_model) representing spatial embeddings.
        :param Xt: A Tensor of shape (batch_size, seq_len,d_model) representing temporal embeddings.
        :param mask: Optional mask for attention.
        :return: A Tensor of shape (batch_size, seq_len, d_model) as the output of the attention mechanism.
        """

        Q = self.W_qs(Xs) + self.W_qt(Xt)
        K = self.W_ks(Xs) + self.W_kt(Xt)
        V = self.W_vs(Xs) + self.W_vt(Xt)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads back into a single output
        output = self.combine_heads(attn_output)
        return output


class DiffTBlock(nn.Module):
    """
    DiffTBlock is a building block that integrates Time-aware Multi-Head Self-Attention (TMSA)
    and a feed-forward neural network (MLP) with layer normalization. It processes spatial and
    temporal embeddings, suitable for tasks like video analysis or temporal sequence modeling.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        """
        :param hidden_size: Dimensionality of the input embeddings. It must be divisible by num_heads.
        :param num_heads: Number of attention heads for the TMSA module.
        :param mlp_ratio: Ratio of the hidden size of the MLP to the input hidden size. Default is 4.0.
            It will multiply the hidden_size by mlp_ratio.
        """
        super().__init__(**block_kwargs)
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.tmsa = TMSA(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(in_features=int(hidden_size * mlp_ratio), out_features=hidden_size),
        )

    def forward(self, x, c):
        """
        :param x: Input tensor of shape (batch_size, seq_len, hidden_size).
        :param c: Context tensor for attention of size (batch_size, hidden_size), one for each input.
            It's usually a combination of label and temporal embedding.
        :return: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        seq_len = x.shape[1]
        c = c.unsqueeze(1).repeat(1, seq_len, 1)
        x = self.norm1(self.tmsa(x, c)) + x
        x = self.norm2(self.mlp(x)) + x
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiffiT. It's necessary to transform the tensor to the necessary shape to unpatchify it correctly.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        """
        :param hidden_size: dimensionality of the input embeddings.
        :param patch_size: patch size used to patch the input image.
        :param out_channels: number of output channels the image should have.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, seq_len, hidden_size).
        :return: Output tensor of shape (batch_size,seq_len, patch_size * patch_size * out_channels).
        """
        x = self.linear(x)
        return x


################## Tests ##########################
TESTING = False
if TESTING and __name__ == '__main__':
    print("Testing TMSA")
    timestep_encoder = TimestepEmbedder(64)
    times = torch.randint(1, 10, (5,))
    embedded_times = timestep_encoder(times)

    sequence = torch.rand(5, 30, 64)  # (batch_size, d_model, img_size)
    print("Embedded times shape: ", embedded_times.unsqueeze(1).repeat(1, 30, 1).shape)
    print("Sequence shape: ", sequence.size())
    tmsa = TMSA(64, num_heads=4)
    processed_sequence = tmsa(sequence, embedded_times.unsqueeze(1).repeat(1, 30, 1))
    print("Processed sequence shape: ", processed_sequence.size())
    print("-" * 30)
    print("Testing DiffiTBlock")
    diffit_block = DiffTBlock(
        hidden_size=64,
        num_heads=4
    )
    print("Embedded times shape: ", embedded_times.shape)
    print("Sequence shape: ", sequence.size())
    diffit_sequence = diffit_block(sequence, embedded_times)
    print("Processed sequence shape: ", diffit_sequence.size())
