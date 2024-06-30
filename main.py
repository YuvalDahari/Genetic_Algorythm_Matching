import random
import matplotlib.pyplot as plt
import numpy as np

# Global variables for preferences, population size, genetic algorithm parameters
DATA_FILE_NAME = "GA_input.txt"
MEN_FAVOR = {}
WOMEN_FAVOR = {}
POPULATION_SIZE = 100
POPULATION = []
COUPLES_AMOUNT = 30
GENERATION_NUM = 180
SELECTION_RATE = 0.2
MUTATION_RATE = 0.05
PLATEAU_GENERATION = 10
FITNESS_VALUE = 3.33    # 100 / COUPLES_AMOUNT = 3.33
LOCAL_MAX_RATE = 90
CALLS_FITNESS_FUNCTION = 0
VARIANCE_THRESHOLD = 5

# Graph' vars
ITERATION_AMOUNT_GRAPH_UPDATES = 6
PERFORMANCE_OVER_GENERATION = {
    "generations": [],
    "max_fitness": [],
    "min_fitness": [],
    "avg_fitness": []
}
OPT_SOLUTION = (0, 0)
MUTATION_ROUND = {
    "generations": [],
    "location": []
}


def update_mutation_info(generation):
    """
    Update the mutation generation and his min fitness
    """
    global MUTATION_ROUND
    min_fitness = POPULATION[0][0]
    for matching in POPULATION:
        if matching[0] < min_fitness:
            min_fitness = matching[0]
    MUTATION_ROUND["generations"].append(generation)
    MUTATION_ROUND["location"].append(1)


def show_performance_graph():
    """
    Create graph that shows the performance over the generation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    fig.canvas.manager.set_window_title('Genetic algorthm of matching')
    bar_width = 1
    r1 = np.array(PERFORMANCE_OVER_GENERATION["generations"])
    r2 = r1 + bar_width

    ax1.plot(PERFORMANCE_OVER_GENERATION["generations"], PERFORMANCE_OVER_GENERATION["max_fitness"], marker='o',
             label='Max Fitness', color='blue', zorder=2)
    ax1.plot(PERFORMANCE_OVER_GENERATION["generations"], PERFORMANCE_OVER_GENERATION["avg_fitness"], marker='o',
             label='Average Fitness', color='orange', zorder=2)
    ax1.plot(PERFORMANCE_OVER_GENERATION["generations"], PERFORMANCE_OVER_GENERATION["min_fitness"], marker='o',
             label='Min Fitness', color='green', zorder=2)
    ax1.plot(MUTATION_ROUND["generations"], MUTATION_ROUND["location"], marker='D',
             label='Mutation round', linestyle='', color='red', zorder=2)

    ax1.set_title('Fitness Values Over Generations')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness Values')
    ax1.legend(loc='upper left')

    ax1.bar(r1, PERFORMANCE_OVER_GENERATION["max_fitness"], color='blue', width=bar_width, edgecolor='grey',
            label='Max Fitness (Bar)', alpha=0.5, zorder=1)
    ax1.bar(r2, PERFORMANCE_OVER_GENERATION["min_fitness"], color='green', width=bar_width, edgecolor='grey',
            label='Min Fitness (Bar)', alpha=0.5, zorder=1)

    ax1.set_xlim(r1[0] - bar_width, r1[-1] + 2 * bar_width)
    ax1.set_xticks(r1 + bar_width / 2)
    ax1.set_xticklabels(PERFORMANCE_OVER_GENERATION["generations"])

    ax2.axis('off')
    table_data = [(f"{x}", f"{y}") for x, y in OPT_SOLUTION[1]]

    # Create table
    table = ax2.table(cellText=table_data, colLabels=['MEN', 'WOMEN'], loc='center', cellLoc='center',
                      colColours=['cyan'] * 2)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.8, 0.8)
    table_title = "Solution"
    ax2.text(0.5, 1, table_title, ha='center', fontsize=12)
    score = "{:.2f}".format(OPT_SOLUTION[0])
    opt_score = "\nScore: " + str(score + "\n")
    ax2.text(0.5, 0.93, opt_score, ha='center', fontsize=10)
    parameters = "Population size: " + str(POPULATION_SIZE) + "    Number of generations: " + str(GENERATION_NUM)
    ax2.text(0.5, 0, parameters, ha='center', fontsize=10)
    parameters = "Number of calls to evaluation function:" + str(CALLS_FITNESS_FUNCTION)
    ax2.text(0.5, -0.05, parameters, ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()


def update_performance_list(generation):
    """
    Update the performance list
    """
    global OPT_SOLUTION
    max_fitness = POPULATION[0][0]
    min_fitness = max_fitness
    sum_fitness = 0
    for matching in POPULATION:
        sum_fitness = sum_fitness + matching[0]
        if matching[0] > max_fitness:
            max_fitness = matching[0]
            OPT_SOLUTION = matching
        elif matching[0] < min_fitness:
            min_fitness = matching[0]
    avg_fitness = sum_fitness / len(POPULATION)
    PERFORMANCE_OVER_GENERATION["generations"].append(generation)
    PERFORMANCE_OVER_GENERATION["max_fitness"].append(max_fitness)
    PERFORMANCE_OVER_GENERATION["min_fitness"].append(min_fitness)
    PERFORMANCE_OVER_GENERATION["avg_fitness"].append(avg_fitness)


def get_difference_between_max_min_solution():
    """
    Check the difference between the high score and the lowest in the population
    """
    max_fitness = POPULATION[0][0]
    min_fitness = max_fitness
    for matching in POPULATION:
        if matching[0] > max_fitness:
            max_fitness = matching[0]
        elif matching[0] < min_fitness:
            min_fitness = matching[0]
    return max_fitness - min_fitness


def mutation_for_everyone():
    """
    Perform mutation for all individuals in the population.
    """
    global POPULATION
    max_fitness = calculate_max_fitness()
    enter = True
    mutations = []
    for score, match in POPULATION:
        # Save the max matching for monotonic improve
        if score == max_fitness and enter:
            mutations.append((score, match))
            enter = False
            continue
        mutated_match = []
        for gene in match:
            if random.random() < MUTATION_RATE:
                # Mutate the gene with a random value
                mutated_gene = (gene[0], random.randint(1, COUPLES_AMOUNT))
            else:
                mutated_gene = gene
            mutated_match.append(mutated_gene)
        mutations.append((0, mutated_match))
    mutations = validation(mutations)
    mutations = do_fitness(mutations)
    POPULATION = mutations


def mutation(children):
    """
    Perform mutation on a given set of children.
    """
    updated_kids = []
    for score, child in children:
        mutated_child = []
        for gene in child:
            if random.random() < MUTATION_RATE:
                # Mutate the gene with a random value
                mutated_gene = (gene[0], random.randint(1, COUPLES_AMOUNT))
            else:
                mutated_gene = gene
            mutated_child.append(mutated_gene)
        updated_kids.append((0, mutated_child))
    return updated_kids


def weighted_choice(population):
    """
    Perform weighted random selection based on fitness scores.
    """
    fitness_scores = [score for score, match in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    return random.choices(population, weights=probabilities)[0]


def validation(children):
    """
    Validate children to ensure they are valid permutations of 1-COUPLES_AMOUNT for men and women.
    If not valid, fix the non-valid pairs.
    """
    valid_children = []
    for score, child in children:
        men, women = zip(*child)
        men_set = set(men)
        women_set = set(women)

        if men_set == set(range(1, COUPLES_AMOUNT + 1)) and women_set == set(range(1, COUPLES_AMOUNT + 1)):
            # If valid, keep the child as is
            valid_children.append((score, child))
        else:
            # If not valid, fix invalid pairs
            missing_men = list(set(range(1, COUPLES_AMOUNT + 1)) - men_set)
            missing_women = list(set(range(1, COUPLES_AMOUNT + 1)) - women_set)

            list_men_in_match = list(men)
            list_women_in_match = list(women)

            for i in range(COUPLES_AMOUNT):
                if list_men_in_match[i] in list_men_in_match[i+1:]:
                    list_men_in_match[i] = missing_men.pop(0)
                if list_women_in_match[i] in list_women_in_match[i+1:]:
                    list_women_in_match[i] = missing_women.pop(0)

            fixed_child = list(zip(list_men_in_match, list_women_in_match))
            valid_children.append((score, fixed_child))

    return valid_children


def crossover(parents):
    """
    Perform crossover between parents to generate children.
    """
    global POPULATION_SIZE
    children_amount = POPULATION_SIZE * (1-SELECTION_RATE)
    children_pop = []

    while children_amount > 0:
        daddy = weighted_choice(parents)[1]
        mummy = weighted_choice(parents)[1]

        # Give another chance to chose different parents (using while could cause infinite loop)
        if mummy == daddy:
            mummy = weighted_choice(parents)[1]

        # Generate crossover point
        crossover_point = random.randint(1, COUPLES_AMOUNT - 1)

        child1 = daddy[:crossover_point] + mummy[crossover_point:]
        child2 = mummy[:crossover_point] + daddy[crossover_point:]

        children_pop.append((0, child1))
        children_pop.append((0, child2))

        children_amount -= 2  # We create 2 children per iteration

    children_pop = mutation(children_pop)
    children_pop = validation(children_pop)
    children_pop = do_fitness(children_pop)

    return children_pop


def sort_population():
    """
    Sort the global population array based on fitness scores in descending order.
    """
    global POPULATION
    POPULATION = sorted(POPULATION, key=lambda x: x[0], reverse=True)


def selection(rate):
    """
    Perform selection of parents based on rate.
    """
    global POPULATION
    global POPULATION_SIZE

    sort_population()
    parents_amount = int(POPULATION_SIZE * rate)

    parents_pop = [POPULATION[i] for i in range(parents_amount)]

    return parents_pop


def sort_preferences(favors):
    """
    Initialize preference dictionaries MEN_FAVOR and WOMEN_FAVOR from file.
    """
    sort_favor = [0] * COUPLES_AMOUNT
    for i in range(len(favors)):
        sort_favor[int(favors[i]) - 1] = i + 1
    return sort_favor


def init_dicts_of_favor(filename):
    """
    Initialize preference dictionaries MEN_FAVOR and WOMEN_FAVOR from file.
    """
    global MEN_FAVOR
    global WOMEN_FAVOR
    num = 0
    with open(filename, 'r') as f:
        for line in f:
            favors = line.strip().split()
            if num < COUPLES_AMOUNT:
                MEN_FAVOR[num] = sort_preferences(favors)
                num += 1
            # There are only COUPLES_AMOUNT men
            else:
                WOMEN_FAVOR[num - COUPLES_AMOUNT] = sort_preferences(favors)
                num += 1


def my_fitness(m):
    """
    Calculate fitness score for a given match.
    """
    global CALLS_FITNESS_FUNCTION
    score = 0
    for man, woman in m:
        man_score = (COUPLES_AMOUNT - int(MEN_FAVOR[man - 1][woman - 1])) * FITNESS_VALUE
        woman_score = (COUPLES_AMOUNT - int(WOMEN_FAVOR[woman - 1][man - 1])) * FITNESS_VALUE
        score += (man_score + woman_score) / 2 + 0.01
    CALLS_FITNESS_FUNCTION += 1
    # Normalized with COUPLES_AMOUNT
    return score / COUPLES_AMOUNT


def do_fitness(pop_array):
    """
    Calculate fitness scores for the entire population array.
    """
    update_pop = []
    for score, match in pop_array:
        fitness_score = my_fitness(match)
        update_pop.append((fitness_score, match))

    return update_pop


def init_population():
    """
    Initialize the initial population with random permutations of couples.
    """
    global POPULATION

    men = list(range(COUPLES_AMOUNT))
    women = list(range(COUPLES_AMOUNT))

    for _ in range(POPULATION_SIZE):
        random.shuffle(men)
        random.shuffle(women)
        pairs = [(men[i] + 1, women[i] + 1) for i in range(COUPLES_AMOUNT)]
        POPULATION.append((0, pairs))

    POPULATION = do_fitness(POPULATION)


def calculate_max_fitness():
    """
    Calculate the maximum fitness score in the current population.
    """
    global POPULATION

    fitness_scores = [score for score, match in POPULATION]
    max_fitness = max(fitness_scores)
    return max_fitness


def run():
    """
    Run the genetic algorithm for a fixed number of generations.
    """
    global POPULATION

    max_fitness_history = []
    # Var for the graph
    generation = GENERATION_NUM

    for current_generation in range(GENERATION_NUM):
        if current_generation % ITERATION_AMOUNT_GRAPH_UPDATES == 0:
            # Graph updates
            update_performance_list(current_generation)

        max_fitness = calculate_max_fitness()
        max_fitness_history.append(max_fitness)

        parents = selection(SELECTION_RATE)
        children = crossover(selection(0.5))
        new_population = parents + children
        POPULATION = list(new_population)

        # Check for too much similarity in the population
        if get_difference_between_max_min_solution() < VARIANCE_THRESHOLD:
            if max_fitness > LOCAL_MAX_RATE:
                generation = current_generation
                break
            else:
                mutation_for_everyone()
                update_mutation_info(current_generation)
                continue

        # Check for plateauing of max fitness
        if len(max_fitness_history) > PLATEAU_GENERATION:
            recent_max_fitness = max_fitness_history[-PLATEAU_GENERATION:]
            if all(fitness == recent_max_fitness[-1] for fitness in recent_max_fitness[:-1]):
                if max_fitness > LOCAL_MAX_RATE:
                    generation = current_generation
                    break
                else:
                    # Reset fitness history
                    max_fitness_history = []
                    mutation_for_everyone()
                    update_mutation_info(current_generation)
    # Graph updates
    update_performance_list(generation)


def print_table(data):
    """
    Print tables with data
    """
    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]
    row_format = "| " + " | ".join(["{:<" + str(width) + "}" for width in col_widths]) + " |"
    separator = "+-" + "-+-".join(["-" * width for width in col_widths]) + "-+"
    print(separator)
    for row in data:
        print(row_format.format(*row))
        print(separator)


def print_info():
    """
    Print info about the running
    """
    print("----------------------------------------------------------")
    print("Number of calls to evaluation function: " + str(CALLS_FITNESS_FUNCTION))
    print("The number of generations: " + str(GENERATION_NUM))
    print("Population size: " + str(POPULATION_SIZE))
    score = "{:.2f}".format(OPT_SOLUTION[0])
    print("The matching score is: " + str(score))
    print("----------------------------------------------------------")
    print("MATCHING SOLUTION")
    print_table([("men", "women")] + OPT_SOLUTION[1])
    print("----------------------------------------------------------")


def main():
    """
    Main function to initialize preferences, population, and run the genetic algorithm.
    """
    file_name = DATA_FILE_NAME
    init_dicts_of_favor(file_name)
    init_population()
    run()
    show_performance_graph()
    print_info()


if __name__ == '__main__':
    main()
