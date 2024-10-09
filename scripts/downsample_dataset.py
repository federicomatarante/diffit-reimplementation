# Example usage in a script
from scripts.utils import ArgumentParser
from training_utils import downsample_dataset

if __name__ == '__main__':
    argument_parser = ArgumentParser()

    # Define command-line arguments
    argument_parser.add_argument('src_path', help_text='Source path of the dataset', type=str)
    argument_parser.add_argument('dst_path', help_text='Destination path for the downsampled dataset', type=str)
    argument_parser.add_argument('new_shape', help_text='New shape as a 2 integeres (e.g. height width)', type=int,
                                 nargs=2)

    # Parse the arguments
    argument_parser.parse_arguments()

    try:
        print("Downsampling dataset...")
        downsample_dataset(argument_parser.args.src_path, argument_parser.args.dst_path, argument_parser.args.new_shape)
    except ValueError as e:
        print(e)
        argument_parser.print_usage()
