import argparse
import sys
import os

import cnn_experiment.in_out.data_iterator as io_di
import cnn_experiment.in_out.names as io_n

def check_exisiting_dir(path):
    path = os.path.realpath(path)
    if not os.path.isdir(path):
       raise argparse.ArgumentTypeError("{} is not a path to a dir".format(path))
    return path


def print_data(input_dir):
    iterator = io_di.data_iterator(input_dir)
    for d, step, x, y_uscore, y_actions in iterator:
        print('dataset: {}, step: {}, X: {}, Y_uscore: {}, Y_actions: {}'
            .format(d, step, x, y_uscore, y_actions))


def main():
    parser = argparse.ArgumentParser(
            description = 'Showcase functionality of data iterator')
    parser.add_argument('-i', '--input_dir',
                        type=check_exisiting_dir,
                        required = True,
                        help='Directory with X and Ys')
    print_data(**vars(parser.parse_args()))


if '__main__' == __name__:
    if sys.version_info > (3, 0):
        main()
    else:
        print('app was tested only with python3')
        exit(1)

