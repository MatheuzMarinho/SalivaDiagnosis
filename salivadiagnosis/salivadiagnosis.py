from argparse import ArgumentParser
from ga import ga
from pso import pso

import pandas as pd
import matplotlib.pyplot as plt
import os
import config_factory


def build_args_parser():
    usage = 'python salivadiagnosis.py\n       ' \
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

    dataset = pd.read_csv(args.dataset_path).dropna(axis='columns')

    config_file = open(args.config_path, 'r')
    config_content = config_file.read()
    config_file.close()

    selected_cols, best_fitness_history, execution_time = run_optimization(args.algorithm, dataset, config_content)
    save_results(selected_cols, best_fitness_history, execution_time, args.out_path)


def run_optimization(algorithm, dataset, config_content):
    result = None
    config = config_factory.build_config(algorithm, config_content)

    if algorithm == 1:
        result = ga.execute(dataset, config)
    elif algorithm == 2:
        result = pso.execute(dataset, config)

    return result


def save_results(selected_cols, best_fitness_history, execution_time, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_path = os.path.join(out_path, "result_metrics.txt")
    file = open(file_path, 'w')
    file.write("Selected cols: " + str(selected_cols) + "\n")
    file.write("Best fitness: " + str(best_fitness_history[-1]) + "\n")
    file.write("Number of Iterations: " + str(len(best_fitness_history)) + "\n")
    file.write("Execution Time: " + str(execution_time) + "\n")
    file.close()

    file_path = os.path.join(out_path, "fitness_history.txt")
    file = open(file_path, 'w')
    for fit in best_fitness_history:
        file.write(str(fit) + "\n")
    file.close()

    plt.plot(best_fitness_history)
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.savefig(out_path + "/fitness_history.png")


if __name__ == '__main__':
    main()
