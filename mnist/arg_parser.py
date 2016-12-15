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
    'ProximalGradientDescent'
]

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('optimizers', nargs='*', choices=ALL_OPTIMIZERS, default=random.choice(ALL_OPTIMIZERS))
    parser.add_argument('--all', action='store_true', help='Runs all optimizers')
    args = parser.parse_args()
    return ALL_OPTIMIZERS if args.all else args.optimizers
