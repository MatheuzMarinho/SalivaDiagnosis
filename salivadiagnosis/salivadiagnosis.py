from argparse import ArgumentParser
from ga import ga
from pso import pso

import pandas as pd
import os
import config_factory


def build_args_parser():
    usage = 'python salivadiagnosis.py --dataset <dataset file>\n       ' \
            'run with --help for arguments descriptions'
    parser = ArgumentParser(description='A python algorithm that optimizes the attributes of the SalivaTec database '
                                        'used to predict patients oral health.', usage=usage)
    parser.add_argument('-d', '--dataset', dest='dataset_path', default='data/dataset_preprocessed.csv',
                        help='Path to the dataset file. Should be in the CSV format.')
    parser.add_argument('-o', '--out', dest='out_path', default='results',
                        help='Path to the directory where all the generated data will be saved.')
    parser.add_argument('-a', '--algorithm', dest='algorithm', type=int, default=1,
                        help='Type of algorithm that will be used to optimize the classification attributes. '
                             'Available types:\n       '
                             '1 - Genetic Algorithm\n       '
                             '2 - Particle Swarm Optimization')
    parser.add_argument('-c', '--config', dest='config_path', required=True,
                        help='Path to the optimization algorithm configuration file. This file should have each '
                             'value in a line and should be in the following order:\n       '
                             '1 - Genetic Algorithm:\n       '
                             '    * Number of individuals\n       '
                             '    * Mutation ratio\n       '
                             '    * Selection type: 1 (Round Robin), 2 (Elitism) or 3 (Roulette)\n       '
                             '    * Number of descendants:\n'
                             '2 - Particle Swarm Optimization:\n       '
                             '    * Population Size\n       '
                             '    * Max number of iterations without improvement\n       '
                             '    * Cognitive coefficient\n       '
                             '    * Social coefficient\n       '
                             '    * Inertia coefficient\n')

    return parser


def main():
    args_parser = build_args_parser()
    args = args_parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    dataset = pd.read_csv(args.dataset_path)

    config_file = open(args.config_path, 'r')
    config_content = config_file.read()
    config_file.close()

    optimization_result = run_optimization(args.algorithm, dataset, config_content)
    # TODO: salvar resultados


def run_optimization(algorithm, dataset, config_content):
    result = None
    config = config_factory.build_config(algorithm, config_content)

    if algorithm == 1:
        result = ga.execute(dataset, config)
    elif algorithm == 2:
        result = pso.execute(dataset, config)

    return result


if __name__ == '__main__':
    main()
