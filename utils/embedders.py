import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    It works in the following way:
     - Embeds scalar timesteps into vector representations.
     - Pass the vector representations through a MLP.
    """

    def __init__(self, hidden_size=512, frequency_embedding_size=256):
        """
        :param hidden_size: hidden size and size of the final reppresentations.
        :param frequency_embedding_size: size of the frequency embeddings. It's the size it uses
        to embed scalar timesteps before feeding into the network.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param t: a 1D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, dim) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        :param t: a tensor of shape (batch_size, timesteps) reppresenting the timesteps to be encoded.
        :return: the encoded timestep representations in a tensor of shape (batch_size, hidden_size).
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    It works in the following way:
     - Embeds class labels into vector representations using an embedding table.
     - Optionally applies label dropout for classifier-free guidance during training.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        """
        :param num_classes: number of classes in the dataset.
        :param hidden_size: size of the embedded representations.
        :param dropout_prob: probability of dropping out labels during training.
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.

        :param labels: a tensor of shape (batch_size,) containing the input labels.
        :param force_drop_ids: optional tensor of shape (batch_size,) to force specific label drops.
        :return: a tensor of shape (batch_size,) with some labels potentially dropped (set to num_classes).
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=False, force_drop_ids=None):
        """
        :param labels: a tensor of shape (batch_size,) containing the input labels.
        :param train: boolean indicating whether the model is in training mode.
        :param force_drop_ids: optional tensor of shape (batch_size,) to force specific label drops.
        :return: the embedded representation of the input labels in a tensor of shape (batch_size, hidden_size).
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


####################### Tests ####################################
TESTING = False
if TESTING and __name__ == '__main__':
    print("Testing TimeStepEmbedder")
    time_embedder = TimestepEmbedder(
        hidden_size=512
    )
    times = torch.randint(0, 64, (5,))
    print("Randomly created Times:", times)
    time_embeddings = time_embedder.forward(times)
    print("Embedded times: ", time_embeddings.size())
    print("-" * 20)
    random_labels = torch.randint(0, 64, (5,))
    print("Randomly created labels:", random_labels)
    label_embedder = LabelEmbedder(num_classes=64, hidden_size=512, dropout_prob=0)
    labels_embeddings = label_embedder.forward(random_labels, train=False, )
    print("Embedded labels: ", labels_embeddings.size())
