import torch
import yaml

from autoencoders.util import instantiate_from_config


class PretrainedAutoEncoder:
    """
    A class to automatically load the pretrained autoencoders from https://github.com/CompVis/latent-diffusion.
    It's necessary to download the correct weights from the GitHub.
    """

    def __init__(self, weights_file, model_config='kl_32x32x4'):
        """

        :param weights_file: the path to the downloaded weights file.
        Careful! Each model has its own right config file to choose.
        :param model_config: the config of the pretrained autoencoders to use. Possible values are:
            'kl_32x32x4', 'kl_16x16x16', 'kl_32x32x4', 'kl_64x64x3'
            The configuration ( with associated model ) decides the shape of the encoded input.
        """
        config_files = {
            'kl_8x8x64': 'autoencoder_kl_8x8x64.yaml',
            'kl_16x16x16': 'autoencoder_kl_16x16x16.yaml',
            'kl_32x32x4': 'autoencoder_kl_32x32x4.yaml',
            'kl_64x64x3': 'autoencoder_kl_64x64x3.yaml',
        }

        assert model_config in config_files.keys(), 'model_config must be one of {}'.format(list(config_files.keys()))

        config_path = r'./autoencoders/configs/' + config_files[model_config]
        # Load the configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Instantiate the model
        model_config = config['model']
        self.model = instantiate_from_config(model_config)
        # Load the pre-trained weights
        state_dict = torch.load(weights_file, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict)

    def encode(self, x):
        """
        Encodes the input x using the pretrained autoencoders.
        :param x: (batch_size, channels, input_size, input_size) tensor of spatial inputs (squared image)
        :return: (batch_size, new_channels, new_input_size, new_input_size) tensor of encoded inputs.
        """
        return self.model.encode(x).sample()

    def decode(self, x):
        """
        Decodes the input x using the pretrained autoencoders.
        :param x: (batch_size, new_channels, new_input_size, new_input_size) tensor of encoded spatial inputs (squared image)
        :return: (batch_size, new_channels, new_input_size, new_input_size) of decoded spatial inputs (squared image)
        """
        return self.model.decode(x)
