import random
import string
import time
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
)

import colorama
colorama.just_fix_windows_console()
# Ensure colorama is initialized for console color support
colorama.init(autoreset=True)

from tabulate import tabulate

# Token to clear the current line in the console
CLEAR_LINE = '\033[K'
# Token to move the cursor up one line in the console
UP = '\033[F'


class GeneticUnit:
    PERFECT_SCORE = 1.
    POSSIBLE_LETTERS = string.ascii_lowercase + ' '

    def __init__(self) -> None:
        self.fitness = 0.
        self.probability = 0.

    def randomize_genes(self, target_length: int) -> None:
        """
        Randomize the genes of the genetic unit with new characters.

        Parameters:
        - target_length (int): The desired length of the genes.

        Returns:
        - None: The randomized genes are assigned to the instance variable `self.genes`.
        """

        # Generate a new string with random characters of the specified length
        self.genes = ''.join(
            GeneticUnit.get_random_char() for _ in range(target_length)
        )

    @classmethod
    def get_random_char(cls) -> str:
        return random.choice(cls.POSSIBLE_LETTERS)

    def calculate_fitness(self, target_phrase: str, target_length: int) -> None:
        """
        Calculate the fitness of the genetic unit based on its genes and a target phrase.

        Parameters:
        - target_phrase (str): The target phrase to compare with.
        - target_length (int): The length of the target phrase.

        Returns:
        - None: The fitness value is assigned to the instance variable `self.fitness`.
        """

        # Initialize the score to 0
        score = 0.

        # Iterate through the corresponding characters of genes and the target phrase
        for x, y in zip(self.genes, target_phrase):
            # Increment the score if the characters match
            if x == y:
                score += 1

        # Calculate the raw fitness as the ratio of matching characters to the target length
        raw_fitness = score / target_length

        # Ensure the fitness is never 0 by adding a small value (0.01)
        # This prevents issues when fitness is used in mathematical operations
        self.fitness = pow(raw_fitness, 2) + 0.01

    def mutate(self, mutation_rate: float) -> None:
        """
        Mutate the genes of the genetic unit based on a mutation rate.

        Parameters:
        - mutation_rate (float): The probability of mutation for each gene.

        Returns:
        - None: The mutated genes are assigned to the instance variable `self.genes`.
        """

        # Create a new string by iterating through each character in the genes
        self.genes = ''.join(
            # Mutate the character with a random character if the mutation condition is met
            GeneticUnit.get_random_char()
            if random.random() < mutation_rate else char
            for char in self.genes
        )

    @staticmethod
    def crossover(a, b, target_length: int) -> 'GeneticUnit':
        """
        Create a new genetic unit by combining genes from two parent units.

        Parameters:
        - a: The first parent genetic unit.
        - b: The second parent genetic unit.
        - target_length (int): The length of the target genes.

        Returns:
        - GeneticUnit: The child genetic unit created by combining genes from parents.
        """

        # Create a new genetic unit for the child
        child = GeneticUnit()

        # Choose a random midpoint for crossover
        midpoint = random.randint(0, target_length)

        # Combine genes from parents based on the chosen midpoint
        child.genes = a[:midpoint] + b[midpoint:]

        # Return the child genetic unit
        return child

    def __getitem__(self, key):
        if isinstance(key, slice):
            # If the key is a slice, apply it to the string value
            return self.genes[key.start:key.stop:key.step]
        elif isinstance(key, int):
            # If the key is an integer, treat it as an index
            return self.genes[key]
        else:
            raise TypeError("Invalid key type. Must be int or slice.")

    def __iter__(self) -> Iterator[str]:
        return self.genes.__iter__()

    def __repr__(self) -> str:
        return self.genes


class GeneticPopulation:
    
    def __init__(self, target_phrase: str, population_count: int, mutation_rate: float) -> None:
        self.target_phrase = target_phrase
        self.target_length = len(target_phrase)
        self.mutation_rate = mutation_rate
        self.population_count = population_count
        self.population = [GeneticUnit() for _ in range(population_count)]

        for unit in self.population:
            unit.randomize_genes(self.target_length)
        
        self.generation = 1

    def _calculate_fitness(self) -> None:
        """
        Calculate the fitness for each genetic unit in the population.

        Returns:
        - None: The fitness values are updated for each genetic unit in the population.
        """

        # Iterate through each genetic unit in the population and calculate its fitness
        for unit in self.population:
            unit.calculate_fitness(self.target_phrase, self.target_length)

    def _evaluate(self) -> Tuple[Optional[GeneticUnit], bool]:
        """
        Evaluate the current population to find the best genetic unit and check for termination.

        Returns:
        - Tuple[Optional[GeneticUnit], bool]: The best genetic unit and a boolean indicating
          whether the termination condition is met.
        """

        # Find the genetic unit with the highest fitness (default to None if population is empty)
        best_unit = max(self.population, key=lambda x: x.fitness, default=None)

        # Check if the best unit has a fitness greater than or equal to the perfect score
        termination_condition_met = best_unit and best_unit.fitness >= GeneticUnit.PERFECT_SCORE

        # Return the best genetic unit and the termination condition flag
        return best_unit, termination_condition_met

    def _generate(self) -> List[GeneticUnit]:
        """
        Generate a new population based on the current population's fitness.

        Returns:
        - List[GeneticUnit]: The new population of genetic units.
        """

        # Calculate the total fitness of the current population
        total_fitness = sum(unit.fitness for unit in self.population)

        # Assign probabilities to each genetic unit based on its fitness
        for unit in self.population:
            unit.probability = unit.fitness / total_fitness

        # Generate a new population by crossover and mutation
        total_prob = sum(unit.probability for unit in self.population)
        new_population = [
            GeneticUnit.crossover(
                self._pick_one(total_prob),
                self._pick_one(total_prob), self.target_length
            ) for _ in range(self.population_count)
        ]

        # Mutate each genetic unit in the new population
        for unit in new_population:
            unit.mutate(self.mutation_rate)

        # Return the new population
        return new_population

    # def _pick_one(self) -> GeneticUnit:
    def _pick_one(self, total_prob: float) -> GeneticUnit:
        """
        Select a random genetic unit from the population based on probabilities.

        Returns:
        - GeneticUnit: The selected genetic unit.
        """

        # Generate a random number between 0 and the total probability
        # random_number = random.uniform(0, sum(unit.probability for unit in self.population))
        random_number = random.uniform(0, total_prob)

        # Use cumulative probability to select the corresponding genetic unit
        cumulative_probability = 0
        for unit in self.population:
            cumulative_probability += unit.probability
            if random_number <= cumulative_probability:
                return unit

        # In case of rounding errors or edge cases, return the last unit
        return self.population[-1]

    def evaluate(self) -> Tuple[GeneticUnit, bool]:
        """
        Evaluate the current population, calculate fitness, and determine termination.

        Returns:
        - Tuple[GeneticUnit, bool]: The best genetic unit and a boolean indicating
          whether the termination condition is met.
        """

        # Calculate fitness for each genetic unit in the population
        self._calculate_fitness()

        # Evaluate the current population and check for termination
        return self._evaluate()

    def evolve(self) -> None:
        """
        Evolve the population by generating a new one and updating the generation count.
        """

        # Generate a new population and update the generation count
        self.population = self._generate()
        self.generation += 1

    def get_average_fitness(self) -> float:
        """
        Calculate the average fitness of the current population.

        Returns:
        - float: The average fitness of the population.
        """

        # Calculate the total fitness of the current population
        total = sum(x.fitness for x in self.population)

        # Return the average fitness if the population is not empty, otherwise, return 0.0
        return total / self.population_count if self.population else 0.0

    # This method is unused in this implementation
    def all_phrases(population, requested_count: int) -> Tuple[str]:
        """
        Unused method. Originally intended to return a tuple of phrases from the population.

        Returns:
        - Tuple[str]: An empty tuple.
        """
        count_limit = min((len(population), requested_count))
        return tuple(x.genes for x in population[:count_limit])


def get_user_input() -> Tuple[str, int, float]:
    """
    Get user input for the target phrase, population count, and mutation rate.

    Returns:
    - Tuple[str, int, float]: A tuple containing the target phrase, population count, and mutation rate.
    """

    # Get a valid target phrase from user input
    target_phrase = get_valid_input(
        'Target phrase', 'hello world',
        lambda x: all(c.isalpha() or c.isspace() for c in x), str
    )

    # Get a valid population count from user input
    population_count = get_valid_input(
        'Population count', 200,
        lambda x: str.isnumeric(x) and int(x) > 0, int
    )

    # Get a valid mutation rate from user input
    mutation_rate = get_valid_input(
        'Mutation rate (0.00 - 1.00)', 0.01,
        lambda x: x.replace('.', '', 1).isdigit(), float
    )

    # Return a tuple containing the user input values
    return target_phrase, population_count, mutation_rate


def get_valid_input(prompt: str, default: Any, validation_func: Callable, cast_func: Callable = float) -> Any:
    """
    Get user input with validation and casting.

    Parameters:
    - prompt (str): The prompt message to display to the user.
    - default (Any): The default value to return if the user input is empty.
    - validation_func (Callable): A function to validate the user input.
    - cast_func (Callable, optional): A function to cast the validated user input. Default is float.

    Returns:
    - Any: The validated and casted user input.
    """

    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()

        # Return default value if user input is empty
        if user_input == '':
            return default

        # Validate user input using the provided validation function
        if validation_func(user_input):
            try:
                # Cast the validated user input using the provided casting function
                return cast_func(user_input)
            except ValueError:
                print('Error: Unable to cast input to the specified type.')

        # If input is invalid, display an error message and prompt again
        print('Please input a valid value.')


def clear_console_lines(num_lines: int = 3) -> None:
    """
    Clear a specified number of lines in the console.

    Parameters:
    - num_lines (int, optional): The number of lines to clear. Default is 3.
    """

    # Iterate through the specified number of lines and clear each line
    for _ in range(num_lines):
        print(UP, end=CLEAR_LINE)


def get_phrase_colorized(phrase: str, target: str) -> str:
    """
    Colorize a phrase based on the matching characters with a target phrase.

    Parameters:
    - phrase (str): The input phrase to colorize.
    - target (str): The target phrase for comparison.

    Returns:
    - str: The colorized phrase as a string.
    """

    # List comprehension to create a list of colorized characters
    colored_chars = [
        (colorama.Fore.GREEN if x == y else colorama.Fore.RED) + x
        for x, y in zip(phrase, target)
    ]

    # Join the colorized characters and reset the color style
    return ''.join(colored_chars) + colorama.Style.RESET_ALL


def display_table(best_unit: Optional[GeneticUnit], target_phrase: str,
                  average: float, generation: int) -> None:
    """
    Display a formatted table with information about the best genetic unit, average fitness, and generation.

    Parameters:
    - best_unit (Optional[GeneticUnit]): The best genetic unit, or None if not available.
    - target_phrase (str): The target phrase for comparison.
    - average (float): The average fitness of the population.
    - generation (int): The current generation number.
    """

    # Check if the best_unit is available
    if best_unit:
        # Colorize the best unit's genes based on the target phrase
        best_phrase_colorized = get_phrase_colorized(best_unit.genes, target_phrase)
        # Format the best fitness and average fitness with color
        best_fitness_formatted = colorama.Fore.YELLOW + f'{best_unit.fitness*100:.3f}' + colorama.Style.RESET_ALL
        average_fitness_formatted = colorama.Fore.CYAN + f'{average*100:.3f}' + colorama.Style.RESET_ALL
    else:
        # Set default values if best_unit is None
        best_phrase_colorized = 'N/A'
        best_fitness_formatted = 'N/A'
        average_fitness_formatted = 'N/A'

    # Create a table using tabulate
    table = tabulate(
        ((
            best_phrase_colorized,
            best_fitness_formatted,
            average_fitness_formatted,
            str(generation),
        ),),
        headers=(
            'Best Phrase', 'Best Fitness', 'Average Fitness', 'Generation',
        ),
        tablefmt='simple'
    )

    # Print the table
    print(table)


def display_plot(average_values: list, best_values: list, generation: int) -> None:
    """
    Display a line plot comparing average and best fitness values over generations.

    Parameters:
    - average_values (list): List of average fitness values for each generation.
    - best_values (list): List of best fitness values for each generation.
    - generation (int): The total number of generations.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Setup Styling
    plt.style.use('seaborn-v0_8-darkgrid')

    # Sample data
    percentages = np.linspace(0, 100, generation)

    # Plotting
    plt.plot(percentages, average_values, label='Average', color='lightblue')
    plt.plot(percentages, best_values, label='Best', color='lightcoral')

    # Adding labels and title
    plt.xlabel('Percentage')
    plt.ylabel('Fitness Values')
    plt.title('Comparison of Average and Best Fitness')

    # Adding legend
    plt.legend()

    # Show the plot
    plt.show()


def main() -> None:
    """
    Run the genetic algorithm to evolve phrases and display results.

    Returns:
    - None: Displays tables and plots during the genetic algorithm execution.
    """

    # Get user input for target phrase, population count, and mutation rate
    target_phrase, population_count, mutation_rate = get_user_input()

    # Record the start time for measuring the total duration
    start_time = time.time()

    # Create a GeneticPopulation instance
    population = GeneticPopulation(target_phrase, population_count, mutation_rate)

    # Lists to store fitness values for plotting
    average_values = []
    best_values = []

    while True:
        # Evaluate the population and get the best unit and termination flag
        best_unit, is_finished = population.evaluate()

        # Calculate and display information about the current generation
        average = population.get_average_fitness()
        display_table(best_unit, target_phrase, average, population.generation)

        # Record fitness values for plotting
        average_values.append(average)
        best_values.append(best_unit.fitness)

        # Check for termination
        if is_finished:
            break

        # Evolve the population for the next generation
        population.evolve()

        # Wait a fraction of a second for better readability of the console output
        time.sleep(0.1)

        # Clear the console lines for the next table print
        clear_console_lines()

    print()  # Just print an empty line
    print(f'Done in {population.generation} generation/s')
    
    end_time = time.time() - start_time
    print(f'Total duration: {colorama.Fore.LIGHTYELLOW_EX}{end_time:.3f} seconds{colorama.Style.RESET_ALL}')

    # Prompt user to show the graph
    user_input = input('Should the graph be shown? [Y]/n: ')
    if 'y' in user_input.lower() or user_input.strip() == '':
        display_plot(average_values, best_values, population.generation)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
