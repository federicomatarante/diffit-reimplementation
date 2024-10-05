import torch
import torch.nn as nn
from timm.layers import PatchEmbed

from diffit import DiffTBlock, FinalLayer
from diffit import TimestepEmbedder
from modeltocopy import get_2d_sincos_pos_embed
from utils.embedders import LabelEmbedder


class Tokenizer(nn.Module):
    """
    Tokenizer module that applies a 2D convolutional layer ( with 3x3 Kernel )  to the input.
    It creates more features maps.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(self, in_channels=3, out_channels=128):
        """
        :param in_channels: number of input channels of the input image.
        :param out_channels: number of output feature maps.
        """
        super(Tokenizer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        """
        Forward pass of the tokenizer.

        :param x: Input tensor of shape (batch_size, in_channels,  height, width).
        :return: Output tensor after applying convolution of shape (batch_size, out_channels, height, width).
        """
        x = self.conv2d(x)
        return x


class Head(nn.Module):
    """
    Head module consisting of Group Normalization followed by a 2D convolutional layer.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(self, in_channels=128, out_channels=3, num_groups=8):
        """
        :param in_channels: number of input channels of the input.
        It must be divisible by num_groups.
        :param out_channels: number of output feature maps.
        :param num_groups: number of groups to divide the input tensor for group normalization.
        """
        super(Head, self).__init__()
        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"
        self.groupNorm = nn.GroupNorm(num_groups, in_channels)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, in_channels, height, width).
        :return: Output tensor of shape (batch_size, channels, height, width).
        """
        x = self.groupNorm(x)
        x = self.conv2d(x)
        return x


class DiffiTResBlock(nn.Module):
    """
    Residual block that applies GroupNorm, SiLU activation, a convolutional layer, and the DiffiT module to the input.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(self, img_size, num_heads=16, patch_size=2, hidden_size=1152, channels=128, num_groups=8):
        """
        :param img_size: size of the input image, assumed squared.
        :param num_heads: number of heads of the DiffiT.
        :param patch_size: patch size for which the image has to be divided in for the vision transformer.
        :param hidden_size: hidden size of the embeddings. Must be divisible by num_heads.
        :param channels: number of channels of the feature map. Must be divisible by num_groups.
        :param num_groups: number of groups to divide the input tensor for group normalization.
        """
        super(DiffiTResBlock, self).__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        assert channels % num_groups == 0, 'patch_size must be divisible by num_groups'
        self.channels = channels
        self.groupNorm = nn.GroupNorm(num_groups, channels)
        self.swish = nn.SiLU()
        self.conv2d = nn.Conv2d(channels, channels, 3, 1, 1)
        self.diffit = DiffTBlock(hidden_size, num_heads)
        self.x_embedder = PatchEmbed(img_size, patch_size, channels, hidden_size, bias=True)

        self.patch_size = patch_size
        self.num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
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

    def forward(self, x, c):
        """
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :param c: Context tensor for attention of size (batch_size, hidden_size), one for each input.
            It's usually a combination of label and temporal embedding.
        :return: Output tensor of shape (batch_size, channels, height, width).
        """
        x = self.conv2d(self.swish(self.groupNorm(x)))
        # Encoding input to pass it to transformer
        x_patched = self.x_embedder(x) + self.pos_embed
        # (N, num_patches, hidden_size), where num_patches = (new_input_size / patch_size ) ** 2
        # Diffit -> Final  -> Unpatchify layer + Residual
        x = self.unpatchify(self.final(self.diffit(x_patched, c))) + x  # (batch_size, channels, height, width)
        return x


class DiffiTSequential(nn.Module):
    """
    Just a sequential version of DiffiTResBlock, so that will be executed sequentially.
    """

    def __init__(self, *blocks):
        """
        :param blocks: a bunch of DiffiTResBlocks.
        """
        super(DiffiTSequential, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def all_equals(n, **kwargs):
        """
        Returns an instance of DiffiTSequential having n blocks with same input parameters.
        :param n: Nr of blocks.
        :param kwargs: parameters of class "DiffiTResBlock".
        :return:
        """
        return DiffiTResBlock(*[DiffiTResBlock(**kwargs) for _ in range(n)])

    def forward(self, x, c):
        """
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :param c: Context tensor for attention of size (batch_size, hidden_size), one for each input.
            It's usually a combination of label and temporal embedding.
        :return: Output tensor of shape (batch_size, channels, height, width).
        """
        for block in self.blocks:
            x = block(x, c)  # Assuming each block takes x and xt
        return x


class ImageDiffiT(nn.Module):  # TODO Maybe implement "learn sigma"
    """
    Diffusion model based on U-Net architecture with a DiffitResBlock backbone.
    More information can be found here:
    https://arxiv.org/pdf/2312.02139
    DiffiT: Diffusion Vision Transformers for Image Generation
    by. Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat
    """

    def __init__(self, img_size, l1=4, l2=4, l3=4, l4=4, patch_size=2, num_classes=1000, class_dropout_prob=0.1,
                 hidden_size=1152, channels=3, hidden_channels=128, num_heads=16, num_groups=8):
        """
        :param l1: number of sequential Diffit Block in the first U-Net level
        :param l2: number of sequential Diffit Block in the first U-Net level
        :param l3: number of sequential Diffit Block in the first U-Net level
        :param l4: number of sequential Diffit Block in the first U-Net level
        :param patch_size: size of the patches to divide the input image.
        :param channels: number of channels in the input image.
        :param hidden_channels: number of hidden channels in the intermediate layers.
            Must be divisible by num_groups.
        :param hidden_size: size of the latent vector representation used inside the network.
            It must be divisible by num_heads.
        :param num_heads: the number of heads in the DiffitBlock transformer.
        :param class_dropout_prob: probability of dropping out class during training.
        :param num_classes: the total number of classes.
        """
        super(ImageDiffiT, self).__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        assert hidden_channels % num_groups == 0, 'hidden_channels must be divisible by num_groups'
        self.num_classes = num_classes
        self.tokenizer = Tokenizer(in_channels=channels, out_channels=hidden_channels)
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        def get_block_params(block_groups, block_img_size):
            """
            Returns the parameters to use in the DiffiTResBlock blocks.
            """
            return {
                "num_heads": num_heads, "patch_size": patch_size, "img_size": block_img_size,
                "hidden_size": hidden_size, "channels": hidden_channels, "num_groups": block_groups
            }

        self.resBlock1 = DiffiTSequential.all_equals(l1, **get_block_params(1, img_size))
        self.downsample_1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.resBlock2 = DiffiTSequential.all_equals(l2, **get_block_params(num_groups, img_size // 2))
        self.downsample_2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.resBlock2 = DiffiTSequential.all_equals(l3, **get_block_params(num_groups, img_size // 4))
        self.downsample_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.resBlock2 = DiffiTSequential.all_equals(l4, **get_block_params(num_groups, img_size // 8))
        self.upsample_1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.resBlock3up = DiffiTSequential.all_equals(l3, **get_block_params(num_groups, img_size // 4))
        self.upsample_2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.resBlock2up = DiffiTSequential.all_equals(l2, **get_block_params(num_groups, img_size // 2))
        self.upsample_3 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.resBlock1Up = DiffiTSequential.all_equals(l1, **get_block_params(1, img_size))
        self.head = Head()
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y):
        """
        Forward pass of the ImageDiffiT model.

        :param x: (batch_size, channels, input_size, input_size) tensor of spatial inputs (squared image)
        :param t: (batch_size,) tensor of diffusion timesteps, one per each image
        :param y: (batch_size,) tensor of class labels, one per each image
        :return: (batch_size, channels, input_size, input_size) tensor of spatial outputs (squared image)
        """
        # Generate embeddings for timesteps and labels
        xt = self.t_embedder(t)  # Timestep embedding: (batch_size, hidden_size)
        xl = self.y_embedder(y)  # Label embedding: (batch_size, hidden_size)
        c = xt + xl  # Combine timestep and label embeddings: (batch_size, hidden_size)

        # Tokenize input image
        x1 = self.tokenizer(x)  # Convert image to feature maps: (batch_size, hidden_channels, input_size, input_size)

        # Encoder (downsampling) path
        x1 = self.resBlock1(x1, c)  # First level of U-Net: (batch_size, hidden_channels, input_size, input_size)
        x2 = self.downsample_1(
            x1)  # Downsample to half resolution: (batch_size, hidden_channels, input_size/2, input_size/2)
        x2 = self.resBlock2(x2, c)  # Second level of U-Net: (batch_size, hidden_channels, input_size/2, input_size/2)
        x3 = self.downsample_2(
            x2)  # Downsample to quarter resolution: (batch_size, hidden_channels, input_size/4, input_size/4)
        x3 = self.resBlock3(x3, c)  # Third level of U-Net: (batch_size, hidden_channels, input_size/4, input_size/4)
        x4 = self.downsample_3(
            x3)  # Downsample to eighth resolution: (batch_size, hidden_channels, input_size/8, input_size/8)
        x4 = self.resBlock4(x4,
                            c)  # Fourth (bottom) level of U-Net: (batch_size, hidden_channels, input_size/8, input_size/8)

        # Decoder (upsampling) path with skip connections
        x3 = x3 + self.upsample_1(
            x4)  # Upsample and add skip connection: (batch_size, hidden_channels, input_size/4, input_size/4)
        x3 = self.resBlock3up(x3,
                              c)  # Process upsampled features: (batch_size, hidden_channels, input_size/4, input_size/4)
        x2 = x2 + self.upsample_2(
            x3)  # Upsample and add skip connection: (batch_size, hidden_channels, input_size/2, input_size/2)
        x2 = self.resBlock2(x2,
                            c)  # Process upsampled features: (batch_size, hidden_channels, input_size/2, input_size/2)
        x1 = x1 + self.upsample_3(
            x2)  # Upsample and add skip connection: (batch_size, hidden_channels, input_size, input_size)
        x1 = self.resBlock1Up(x1,
                              c)  # Process final upsampled features: (batch_size, hidden_channels, input_size, input_size)

        # Generate final output
        x = self.head(x1)  # Convert feature maps to output image: (batch_size, channels, input_size, input_size)

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
    print("Testing Tokenizer")
    tok_input = torch.randn(5, 3, 64, 64)
    tokenizer = Tokenizer(in_channels=3, out_channels=126)
    print("Input shape: ", tok_input.shape)
    print("Output shape: ", tokenizer(tok_input).shape)
    print("-" * 30)
    print("Testing Head")
    head_input = torch.rand(5, 128, 64, 64)
    head = Head(in_channels=128, out_channels=3)
    print("Input shape: ", head_input.shape)
    print("Output shape: ", head(head_input).shape)
    print("-" * 30)
    times_tensor = torch.randint(0, 10, (5,))
    label_tensor = torch.randint(0, 10, (5,))
    print("Testing Image Diffit ")
    diffit_input = torch.rand(5, 3, 64, 64)
    diffit = ImageDiffiT(img_size=64)
    print("Input shape: ", diffit_input.shape)
    print("Output shape: ", diffit(diffit_input, times_tensor, label_tensor).shape)


if __name__ == '__main__':
    main()
