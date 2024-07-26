# Genetic Algorithm Phrase Evolution

This repository contains a Python implementation of a genetic algorithm to evolve a phrase towards a target phrase. The algorithm uses basic genetic operations like mutation and crossover to optimize the population of candidate phrases.

## Features

- **Genetic Algorithm**: Uses selection, crossover, and mutation to evolve phrases.
- **Fitness Calculation**: Compares candidate phrases against a target phrase.
- **User Interaction**: Allows user to input the target phrase, population size, and mutation rate.
- **Visual Feedback**: Displays the best phrase, its fitness, average fitness, and generation count in the console.
- **Plotting**: Optionally plots the fitness values over generations.

## Requirements

- Python 3.7+
- colorama
- tabulate
- matplotlib

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage
Run the script using the following command:

```bash
python main.py
```

Follow the prompts to input the target phrase, population count, and mutation rate.

## Example

When prompted, you can input the following values:

- Target phrase: hello world
- Population count: 200
- Mutation rate: 0.01

The script will then evolve the population towards the target phrase, displaying the progress in the console.

## Script Overview

**`GeneticUnit` Class**

- randomize_genes(target_length): Randomizes the genes of the unit.
- calculate_fitness(target_phrase, target_length): Calculates the fitness of the unit based on a target phrase.
- mutate(mutation_rate): Mutates the genes based on the mutation rate.
- crossover(a, b, target_length): Creates a new unit by combining genes from two parents.

**`GeneticPopulation` Class**

- _calculate_fitness(): Calculates the fitness for each unit in the population.
- _evaluate(): Evaluates the current population to find the best unit and checks for termination.
- _generate(): Generates a new population based on fitness.
- _pick_one(total_prob): Selects a unit from the population based on probability.
- evaluate(): Public method to evaluate the population.
- evolve(): Evolves the population for the next generation.
- get_average_fitness(): Calculates the average fitness of the population.

**Utility Functions**

- get_user_input(): Prompts the user for input.
- get_valid_input(prompt, default, validation_func, cast_func): Validates and casts user input.
- clear_console_lines(num_lines): Clears the specified number of lines in the console.
- get_phrase_colorized(phrase, target): Colorizes the phrase based on matching characters.
- display_table(best_unit, target_phrase, average, generation): Displays a table with the best unit and fitness information.
- display_plot(average_values, best_values, generation): Plots the fitness values over generations.

**`main()` Function**

Runs the genetic algorithm, prompts for user input, evaluates the population, and evolves it until the termination condition is met.

## Contributing

Feel free to fork the repository and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
