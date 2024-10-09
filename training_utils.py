import math
import os

import torch
import yaml
from PIL import Image


def validate_parameters(params: dict, param_types: dict) -> None:
    """
    Validates that the parameters dictionary contains all the required keys and that their types match the expected values.

    :param params: A dictionary containing the parameters loaded from a YAML file.
    :param param_types: A dictionary where keys are parameter names and values are expected types.
    :raises ValueError: If any required parameter is missing.
    :raises TypeError: If a parameter is of the wrong type.
    """
    missing_params = [param for param in param_types if param not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters in YAML: {', '.join(missing_params)}")

    for param, expected_type in param_types.items():
        if not isinstance(params[param], expected_type):
            raise TypeError(
                f"Parameter '{param}' must be of type {expected_type.__name__}, but got {type(params[param]).__name__}.")


def load_parameters(config_file: str, param_types: dict) -> dict:
    """
    Loads and validates parameters from a YAML configuration file.

    :param config_file: Path to the YAML file containing training parameters.
    :param param_types: A dictionary where keys are parameter names and values are expected types.
    :return: A dictionary of parameters loaded from the file.
    :raises ValueError: If required parameters are missing.
    :raises TypeError: If parameters have incorrect types.
    """
    with open(config_file, 'r') as file:
        params = yaml.safe_load(file)

    # Define the expected types for each parameter
    validate_parameters(params, param_types)

    # Ensure test_size is within the valid range (0, 1)
    if not (0 < params['test_size'] < 1):
        raise ValueError("Parameter 'test_size' must be a float between 0 and 1")

    return params


def downsample_dataset(input_dir, output_dir, target_size, verbose=False):
    """
    Downsamples all images in a dataset organized in class folders and saves them to a new dataset.

    :param input_dir: Path to the input dataset directory. Each subfolder represents a class.
    :param output_dir: Path to the output dataset directory where downsampled images will be saved.
    :param target_size: The desired size (width, height) to downsample the images.
    :param verbose: If true, prints progress.

    The function maintains the folder structure from the input directory and saves downsampled images 
    in the corresponding class folders in the output directory.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_dir, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                try:
                    with Image.open(img_path) as img:
                        downsampled_img = img.resize(target_size)
                        output_img_path = os.path.join(output_class_path, img_name)
                        downsampled_img.save(output_img_path)
                        if verbose:
                            print(f"Processed {img_name} and saved to {output_img_path}")
                except Exception as e:
                    print(f"Failed to process {img_name}: {e}")


class NoiseScheduler:
    def __init__(self, diffusion_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = diffusion_steps
        self.BETA_START = beta_start
        self.BETA_END = beta_end

    def beta_comp(self, t):
        return self.BETA_START + (t / self.T) * (self.BETA_END - self.BETA_START)

    def alpha_comp(self, t):
        return 1 - self.beta_comp(t)

    def alpha_hat_comp(self, t):
        return math.prod([self.alpha_comp(j) for j in range(t)])

    def noisify(self, sample, timestep):
        noise = torch.randn(sample.shape, device=sample.device, requires_grad=True)
        noise_sample = []
        for i in range(len(timestep)):
            alpha_hat = self.alpha_hat_comp(timestep[i])
            noise_sample.append(
                (math.sqrt(alpha_hat) * sample[i]) + (math.sqrt(1 - alpha_hat) * noise[i])
            )
        noise_sample = torch.stack(noise_sample, dim=0)
        return noise, noise_sample
