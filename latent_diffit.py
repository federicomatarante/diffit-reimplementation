import os.path

import torch
from timm.layers import PatchEmbed
from torch import nn

from autoencoders.pretrained_autoencoder import PretrainedAutoEncoder
from diffit import DiffTBlock, FinalLayer
from utils.embedders import TimestepEmbedder, LabelEmbedder
from utils.positional_embeddings import get_2d_sincos_pos_embed


class LatentDiffiT(nn.Module):
    """
    Diffusion model with a Transformer backbone which uses the latent space of the model.
    The image is encoded and decoded by a pretrained variational autoencoders, and processed by batch_size layers of the
    DiffiT Block.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(
            self,
            autoencoder,
            encode_size=32,
            patch_size=2,
            channels=3,  # TODO implement classifier-free guidance
            hidden_size=1152,
            depth=30,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
    ):
        """
        :param autoencoder: the autoencoders model used to encode and decode the input. It must have the following methods:
            - encode(input) returns a tensor of shape [batch_size, seq_len, hidden_size]
            - decode(input): returns a tensor of shape [batch_size, img_size, img_size]
            Recommended to use autoencoders.pretraiend_autoencoder.PretrainedAutoEncoder
        :param encode_size: height and width of the input image - assumed to be square - after the encoding.
        :param patch_size: size of the patches to divide the input image.
        :param channels: number of channels in the input image.
        :param hidden_size: size of the latent vector representation used inside the network.
            It must be divisible by num_heads.
        :param depth: number of DiffitBlocks.
        :param num_heads: the number of heads in the DiffitBlock transformer.
        :param mlp_ratio: Ratio of the hidden size of the MLP to the input hidden size in the DiffitBlock.
        :param class_dropout_prob: probability of dropping out class during training.
        :param num_classes: the total number of classes.
        """
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'

        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.autoencoder = autoencoder
        self.x_embedder = PatchEmbed(encode_size, patch_size, channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiffTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final = FinalLayer(hidden_size, patch_size, self.channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x):
        """
        Transforms a batch of series of patches to a batch of unpatched images.
        :param x: Input tensor of size (batch_size, T, patch_size**2 * C)
        :return: Output tensor of size (batch_size, H, W, C)
        """
        c = self.channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        :param x: (batch_size, channels, input_size, input_size) tensor of spatial inputs (squared image)
        :param t: (batch_size,) tensor of diffusion timesteps, one per each image.
        :param y: (batch_size,) tensor of class labels, one per each image.
        :return (batch_size, channels, input_size, input_size) tensor of spatial inputs (squared image)
        """
        with torch.no_grad():
            # Encoding the image
            x = self.autoencoder.encode(x)  # (batch_size, new_channels, new_input_size, new_input_size)

        # Patchify the latent representation and adding positional embedding to patches
        x = self.x_embedder(x) + self.pos_embed
        # (batch_size, num_patches, hidden_size), where num_patches = (new_input_size / patch_size ) ** 2
        # Creating embedding of time steps
        t = self.t_embedder(t)  # (batch_size, hidden_size)
        # Creating embedding of class labels
        y = self.y_embedder(y, self.training)  # (batch_size, hidden_size)
        # Summing time and label embedding
        c = t + y  # (batch_size, hidden_size)
        # Passing through all the DiffiT blocks
        for block in self.blocks:
            x = block(x, c)  # (batch_size, num_patches, hidden_size)
        # Passing to final layer to allow correct unpatchify module
        x = self.final(x)  # ( batch_size, num_patches, patch_size**2 * new_channels )
        # Transforming patches into actual latent representations
        x = self.unpatchify(x)  # (batch_size, new_channels, new_input_size, new_input_size )
        with torch.no_grad():
            # Decoding the latent representations into image
            x = self.autoencoder.decode(x)  # (batch_size, channels, input_size)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):  # TODO to understands and adapt to Diffit
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


################## Tests ##########################
def main(testing=False):
    if not testing:
        return
    with torch.no_grad():
        image_tensor = torch.rand((5, 3, 256, 256))

        times_tensor = torch.randint(0, 10, (1,))
        label_tensor = torch.randint(0, 10, (1,))
        dit_model = LatentDiffiT(autoencoder=PretrainedAutoEncoder('model.ckpt'), channels=4)
        print("Image shape: ", image_tensor.shape)
        print("Label shape: ", label_tensor.shape)
        print("Time shape: ", times_tensor.shape)
        x = dit_model(image_tensor, times_tensor, label_tensor)
        print("Output shape: ", x.shape)


if __name__ == '__main__':
    main()
