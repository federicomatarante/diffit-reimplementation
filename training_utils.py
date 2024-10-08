import os
from PIL import Image


def downsample_dataset(input_dir, output_dir, target_size,verbose=False):
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
