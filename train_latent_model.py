import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from autoencoders.pretrained_autoencoder import PretrainedAutoEncoder
from latent_diffit import LatentDiffiT
from scripts.utils import ArgumentParser
from training import DiffiTTrainer
from training_utils import load_parameters, NoiseScheduler


def train_latent_model(config_file: str = 'train_parameters.yaml') -> None:
    """
    Trains a latent model based on parameters specified in a YAML configuration file.

    :param config_file: Path to the YAML file containing training parameters.
    """
    param_types = {
        # Dataset settings
        'img_size': int,  # Size of the input images (e.g., 256)
        'save_folder': str,  # Path to save model checkpoints and results
        'dataset_folder': str,  # Path to the dataset folder

        # Training settings
        'epochs': int,  # Number of epochs for training
        'batch_size': int,  # Batch size for training
        'learning_rate': (float, int),  # Learning rate for the optimizer
        'test_size': float,  # Fraction of data used for testing (between 0 and 1)
        'random_seed': int,  # Random seed for reproducibility
        'loss_function': str,  # Loss function (e.g., 'MSELoss')

        # Model settings
        'autoencoder_checkpoint': str,  # Path to the pre-trained autoencoder checkpoint
        'channels': int,  # Number of input/output channels
        'patch_size': int,  # Patch size for the model (used in transformer-based models)
        'hidden_size': int,  # Size of the latent sapce in the model
        'depth': int,  # Number of layers in the model
        'num_heads': int,  # Number of attention heads in transformer-based models
        'mlp_ratio': (float, int),  # Ratio for MLP hidden dimension in transformer models
        'class_dropout_prob': float,  # Dropout probability for classification layers
        'num_classes': int,  # Number of output classes for classification

        # Diffusion settings
        'diffusion_steps': int,  # Number of the diffusion steps in the diffusion process
        'beta_start': float,  # Beta start for noise generation
        'beta_end': float,  # Bend end for noise generation
    }

    params = load_parameters(config_file, param_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using: {device}")
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    save_folder = params['save_folder']
    dataset_folder = params['dataset_folder']
    try:
        os.makedirs(save_folder)
    except OSError as e1:
        print(f"Creation of the directory {save_folder} failed")
        print(f"Error: {e1}")

    model = LatentDiffiT(
        autoencoder=PretrainedAutoEncoder(params['autoencoder_checkpoint']),
        encode_size=params['img_size'] // 8,
        patch_size=params['patch_size'],
        channels=params['channels'],
        hidden_size=params['hidden_size'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        mlp_ratio=params['mlp_ratio'],
        class_dropout_prob=params['class_dropout_prob'],
        num_classes=params['num_classes'],
    )
    model = model.to(device)
    loss = getattr(nn, params['loss_function'])()
    optimizer = optim.Adam(model.parameters(), params['learning_rate'])

    torch.manual_seed(params['random_seed'])

    dataset = datasets.ImageFolder(
        root=dataset_folder,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset, dataset.targets, test_size=params['test_size'], random_state=params['random_seed']
    )
    train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    noise_scheduler = NoiseScheduler(
        diffusion_steps=params['diffusion_steps'],
        beta_start=params['beta_start'],
        beta_end=params['beta_end'],
    )

    trainer = DiffiTTrainer(train_dataloader=train_dataloader,
                            valid_dataloader=test_dataloader,
                            model=model,
                            optimizer=optimizer,
                            loss_function=loss,
                            device=device,
                            save_folder=save_folder,
                            batch_size=params['batch_size'],
                            num_epochs=params['epochs'],
                            noise_scheduler=noise_scheduler
                            )
    trainer.train_and_validate()


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('config_file', help_text='YAML configuration file path', type=str)
    argument_parser.parse_arguments()

    try:
        train_latent_model(argument_parser.args.config_file)
    except ValueError as e:
        print(e)
        argument_parser.print_usage()
