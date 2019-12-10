from random import sample, randint, choices, choice, shuffle
from mlp import mlp
import time
import heapq


def execute(dataset, config):
    cols = list(dataset)[:-1]
    population = __init_population(cols, config.num_individuals)

    unimproved_iterations_limit = 30

    best_fitness = None
    best_fitness_history = []
    best_individual = None
    unimproved_iterations = 0
    num_iterations = 0
    execution_time = 0

    print('########## Optimization ##########\n')
    while unimproved_iterations < unimproved_iterations_limit:
        print('********** Iteration ' + str(num_iterations + 1) + ' **********')

        iter_start = time.time()
        num_iterations += 1

        population = __reproduction(population, config, dataset)
        population = __selection(population, config, dataset)
        best_individual_candidate = __get_best_individuals(population, 1, dataset)[0]
        best_fitness_candidate = __get_fitness(best_individual_candidate, dataset)

        if best_fitness is None or best_fitness_candidate > best_fitness:
            best_fitness = best_fitness_candidate
            best_individual = best_individual_candidate
            unimproved_iterations = 0

            print('New best individual with fitness ' + str(best_fitness))
        else:
            unimproved_iterations += 1

            print('No improvement in this iteration. ' +
                  'Number of unimproved iterations: ' +
                  str(unimproved_iterations) + '/' + str(unimproved_iterations_limit)
                  )

        best_fitness_history.append(best_fitness)

        iter_time = time.time() - iter_start
        execution_time += iter_time
        print('Execution time: ' + str(execution_time) + ' s')
        print('*********************************\n')

    print('##################################')

    return best_individual, best_fitness_history, execution_time


def __init_population(cols, num_individuals):
    population = []
    num_cols = len(cols)

    for i in range(num_individuals):
        selected_indices = sample(range(num_cols), randint(0, num_cols))

        for j in range(num_cols):
            if j in selected_indices:
                population.append((cols[j], True))
            else:
                population.append((cols[j], False))

    return population


def __reproduction(population, config, dataset):
    new_population = []

    weights = __get_population_weights(population, dataset)

    while len(new_population) < config.num_individuals * config.num_descendants:
        parents = __choices_no_replacement(population, weights, 2)

        for i in range(config.num_descendants):
            new_population.append(__generate_descendant(parents, config.mutation_ratio))

    new_population += population
    return new_population


def __get_population_weights(population, dataset):
    all_fitness = [__get_fitness(i, dataset) for i in population]
    worst_fitness = max(all_fitness)
    weights = [1 - (curr_fitness / worst_fitness) for curr_fitness in all_fitness]
    return weights


def __get_fitness(individual, dataset):
    return mlp.train(individual, dataset)


def __choices_no_replacement(population, weights, num_choices):
    result = []
    indices = list(range(len(population)))
    weights_copy = weights.copy()

    while num_choices > 0:
        index = choices(population=indices, weights=weights_copy, k=1)[0]
        result.append(population[index].copy())
        i = indices.index(index)
        del indices[i]
        del weights_copy[i]
        num_choices -= 1

    return result


def __generate_descendant(parents, mutation_ratio):
    num_cols = len(parents[0])
    crossover_index = choice(list(range(1, num_cols)))
    parent_1_frag = parents[0][0:crossover_index]
    parent_2_frag = parents[1][crossover_index:]
    descendant = parent_1_frag + parent_2_frag
    __mutate(descendant, mutation_ratio)

    return descendant


def __mutate(descendant, mutation_ratio):
    if mutation_ratio > 0:
        num_cols = len(descendant[0])
        num_cols_to_mutate = int(num_cols * mutation_ratio)
        indices_to_mutate = sample(range(num_cols), num_cols_to_mutate)

        for i in indices_to_mutate:
            if choice([True, False]):
                col_name, col_value = descendant[i]
                descendant[i] = (col_name, not col_value)


def __selection(population, config, dataset):
    if config.selection_type == 1:
        return __round_selection(population, config.num_individuals, config.num_descendants, dataset)
    elif config.selection_type == 2:
        return __elitist_selection(population, config.num_individuals, dataset)
    elif config.selection_type == 3:
        return __roulette_selection(population, config.num_individuals, dataset)


def __round_selection(population, num_individuals, num_descendants, dataset):
    shuffle(population)
    round_size = num_descendants + 2

    new_population = []
    for i in range(0, num_individuals):
        population_slice = population[i * round_size:round_size * (i + 1)]
        new_population.append(__get_best_individuals(population_slice, 1, dataset)[0])

    return new_population


def __get_best_individuals(population, num_individuals, dataset):
    all_fitness = [__get_fitness(individual, dataset) for individual in population]
    best_fitness = heapq.nlargest(num_individuals, all_fitness)
    best_fitness_indices = [all_fitness.index(curr_fitness) for curr_fitness in best_fitness]
    best_individuals = [population[i] for i in best_fitness_indices]

    return best_individuals


def __elitist_selection(population, num_individuals, dataset):
    new_population = __get_best_individuals(population, num_individuals, dataset)
    return new_population


def __roulette_selection(population, num_individuals, dataset):
    weights = __get_population_weights(population, dataset)
    return __choices_no_replacement(population, weights, num_individuals)


