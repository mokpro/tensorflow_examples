from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import random

ALL_OPTIMIZERS = [
    'GradientDescent',
    'Adagrad',
    'Adadelta',
    'ProximalAdagrad',
    'ProximalGradientDescent',
    'Adam'
]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'optimizers',
        nargs='*',
        choices=ALL_OPTIMIZERS,
        default=random.choice(ALL_OPTIMIZERS))
    parser.add_argument(
        '--all',
        action='store_true',
        help='Runs all optimizers')
    args = parser.parse_args()
    if args.all:
        final_optimizers = ALL_OPTIMIZERS
    elif isinstance(args.optimizers, list):
        final_optimizers = args.optimizers
    else:
        final_optimizers = [args.optimizers]

    return final_optimizers
