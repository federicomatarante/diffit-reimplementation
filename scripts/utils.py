import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Command-line argument parser.')
        self.args = None

    def add_argument(self, name, help_text, type=str, nargs=None):
        """Add an argument to the parser."""
        self.parser.add_argument(name, help=help_text, type=type, nargs=nargs)

    def parse_arguments(self):
        """Parse the command-line arguments."""
        try:
            self.args = self.parser.parse_args()
        except SystemExit as e:
            # If there is an error in parsing, print usage and exit
            self.print_usage()
            raise e

    def print_usage(self):
        """Print the usage of the parser with types."""
        print("Usage:")
        print(self.parser.format_usage())
        print("Argument types:")
        for action in self.parser._actions:
            if action.dest != 'help':  # Skip help action
                print(f"  {action.dest} ({action.type.__name__}) - {action.help}")

