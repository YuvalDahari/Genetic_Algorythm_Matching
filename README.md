# Genetic Algorithm for Matching Problem
This project implements a genetic algorithm to solve a matching problem where men and women have preferences for each other. The goal is to find an optimal matching that maximizes the overall satisfaction based on given preferences.

## Features
- Customizable population size, selection rate, mutation rate, and number of generations
- Handles preferences for men and women from an input file
- Visualizes the performance of the algorithm over generations
- Displays the final matching solution and its fitness score
## Installation
### Executable Version
1. Download the executable:
  Download the main.exe file from the releases section.

2. Prepare the input file:
  Create the file named GA_input.txt with the preferences for men and women. Each line should contain the preferences for one individual, with preferences separated by spaces. Place this file in the same directory as main.exe.

3. Run the executable:

  Double-click main.exe to run the genetic algorithm.

### Python Script Version
1. Clone the repository:
```bash
git clone https://github.com/yourusername/genetic-algorithm-matching.git
cd genetic-algorithm-matching
```
2. Install dependencies:
  This project requires matplotlib and numpy. You can install them using pip:
  ```bash
  pip install matplotlib numpy
  ```
3. Prepare the input file:

  make shure you have the file named GA_input.txt with the preferences for men and women. Each line should contain the preferences for one individual, with preferences separated by spaces.

4. Run the script:

  Execute the main script to run the genetic algorithm:
    ```bash
    python main.py
    ```
## results:

The script will display performance graphs and print the final matching solution to the console.also you can check results.pdf for more details.
The final matching solution is displayed in a table format with the fitness score and details about the number of evaluation function calls, population size, and generations.
### running examples results
<p align="center">
  <img src="https://github.com/YuvalDahari/Genetic_Algorythm_Matching/blob/main/first.jpg?raw=true" alt="level 1" width="45%"/>
  <img src="https://github.com/YuvalDahari/Genetic_Algorythm_Matching/blob/main/%D7%AA%D7%9E%D7%95%D7%A0%D7%94%20%D7%A9%D7%9C%20WhatsApp%E2%80%8F%202024-06-30%20%D7%91%D7%A9%D7%A2%D7%94%2003.20.48_00af1fb3.jpg?raw=true" alt="level 2" width="45%"/>
</p>
<p align="center">
  <img src="https://github.com/YuvalDahari/Genetic_Algorythm_Matching/blob/main/%D7%AA%D7%9E%D7%95%D7%A0%D7%94%20%D7%A9%D7%9C%20WhatsApp%E2%80%8F%202024-06-30%20%D7%91%D7%A9%D7%A2%D7%94%2003.21.02_97d40393.jpg?raw=true" alt="level 3" width="45%"/>
</p>

## Configuration
You can configure various parameters of the genetic algorithm by modifying the global variables in the script:

**POPULATION_SIZE**: Number of individuals in the population.
**SELECTION_RATE**: Proportion of the population selected for reproduction.
**MUTATION_RATE**: Probability of mutation for each gene.
**GENERATION_NUM**: Number of generations to run the algorithm.
**COUPLES_AMOUNT**: Number of couples (men and women) to match.
## Algorithm Details
### Initialization
**Preferences**: Read from GA_input.txt to initialize MEN_FAVOR and WOMEN_FAVOR dictionaries.
**Population**: Randomly generated initial population of matchings.
### Genetic Operators
**Selection**: Select parents based on fitness scores.
**Crossover**: Combine genes from two parents to create children.
**Mutation**: Randomly alter genes with a specified probability.
### Fitness Calculation
**Fitness Function**: Calculate fitness score based on the preferences of men and women.
**Evaluation**: Assign fitness scores to all individuals in the population.
### Execution
- Run the genetic algorithm for a fixed number of generations.
- Track performance over generations.
- Apply mutation if the population converges or plateaus.
- Performance Graphs
- The script generates performance graphs showing the maximum, average, and minimum fitness values over generations. It also highlights mutation rounds.
