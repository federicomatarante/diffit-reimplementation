import torch
from timm.layers import PatchEmbed
from torch import nn

from autoencoders.pretrained_autoencoder import PretrainedAutoEncoder
from diffit import DiffTBlock, FinalLayer
from utils.embedders import TimestepEmbedder, LabelEmbedder
from utils.positional_embeddings import get_2d_sincos_pos_embed


class LatentDiffiT(nn.Module):  # TODO maybe we can implement "learn sigma"
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
            channels=3,
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

        self.num_classes = num_classes
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
        :return: (batch_size, channels, input_size, input_size) tensor of spatial inputs (squared image)
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

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of the model with a 3-channels classifiers free guidance.
        Basically it works in the following way:
        - The input x is repeated twice creating a new batch.
        - The input t is repeated twice creating a new batch.
        - The input y is concatenated to a batch of the same size in which each element is a "null" label ( which in
        this case is "null_classes" ).
        The model will predict the noise of each image 2 times: one guided ( when there is the label ) and
        one not guided ( when the label is "null" ).
        The process is applied only to the first 3 channels.

        :param cfg_scale: the classifier-free-guidance scale. It's a parameter. The highest the number is, the more the
         conditioning has importance.
        :param x: (batch_size, channels, input_size, input_size) tensor of spatial inputs (squared image)
        :param t: (batch_size,) tensor of diffusion timesteps, one per each image.
        :param y: (batch_size,) tensor of class labels, one per each image.
        :return: (batch_size*2, channels, input_size, input_size) tensor of spatial inputs (squared image)
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        combined = torch.cat([x, x], dim=0)
        combined_times = torch.cat([t, t], dim=0)
        null_labels = torch.full((x.shape[0],),
                                 self.num_classes)  # "self.num_classes" is the special class for "no class"
        combined_labels = torch.cat([y, null_labels], dim=0)
        model_out = self.forward(combined, combined_times, combined_labels)
        # Eps: first 3 channels
        # Rest: remaining channels ( usually none )
        eps, rest = model_out[:, :3], model_out[:, 3:]
        # This is only about the first 3 channels
        # cond_eps: noise generated conditionally
        # uncond_eps: noise generated unconditionally.
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # The noise is combined between the 2 using the weight "cfg scale"
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # The batch is "doubled" going back to the original shape TODO why is it useful? Can't we just use half?
        eps = torch.cat([half_eps, half_eps], dim=0)
        # The three channels are combined with the rest - untouched
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
