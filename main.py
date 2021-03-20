"""
The main module of DAFEST project.

ADAFEST is an abbreviation:  'A Data-Driven Approach to Estimating / Evaluating Software Testability'

The full version of source code will be available
as soon as the relevant paper(s) are published.

"""


class Main():
    """Welcome to project ADAFEST
    This file contains the main script
    """

    @classmethod
    def print_welcome(cls, name) -> None:
        """
        Print welcome message
        :param name:
        :return:
        """
        print(f'Welcome to the project {name}.')


# Main driver
if __name__ == '__main__':
    Main.print_welcome('ADAFEST')
