#!/usr/bin/python3

from copy import copy
import heapq
import itertools
from random import randrange
import time

import numpy as np
from PyQt6.QtCore import QLineF, QPointF
from TSPClasses import *


class TSPSolver:
    """TSP Solver."""

    def __init__(self, gui_view):
        """Initialize the TSPSolver."""
        self._scenario = None
        self.geneticSolver = GeneticSolver(scenario=self._scenario)

    def setupWithScenario(self, scenario):
        """Set up with scenario."""
        self._scenario = scenario

    def defaultRandomTour(self, time_allowance=60.0):
        """Compute a solution to the TSP problem for the scenario using a random tour.

        Can be used to find an initial BSSF.

        Returns:
            results dictionary for GUI that contains three ints: cost of best solution,
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this
            algorithm
        """
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results["cost"] = bssf.cost if foundTour else math.inf
        results["time"] = end_time - start_time
        results["count"] = count
        results["soln"] = bssf
        results["max"] = None
        results["total"] = None
        results["pruned"] = None
        return results

    def greedy(self, time_allowance=60.0):
        """Compute a solution to the TSP problem for the scenario using a greedy algorithm.

        This algorithm will always look for the closest city to the current city and add it to the route.

        Returns:
            results dictionary for GUI that contains three ints: cost of best solution,
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this
            algorithm
        """
        return greedyTSP(self._scenario.getCities(), time_allowance=time_allowance)

    def branchAndBound(self, time_allowance=60.0):
        """Compute a solution to the TSP problem for the scenario using a branch-and-bound algorithm.

        Returns:
            results dictionary for GUI that contains three ints: cost of best solution,
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this
            algorithm
        """
        tree = BranchAndBound(scenario=self._scenario, time_allowance=time_allowance)
        return tree.solve()

    def fancy(self, time_allowance=60.0):
        """Compute a solution to the TSP problem for the scenario using a genetic algorithm.

        Returns:
            results dictionary for GUI that contains three ints: cost of best solution,
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this
            algorithm
        """
        self.geneticSolver._scenario = self._scenario
        self.geneticSolver._timeAllowance = time_allowance
        return self.geneticSolver.solve()


# GENETIC ALGORITHM
class GeneticSolver:
    """Genetic algorithm solver for the traveling salesperson problem."""

    # Selection types
    SELECTION_ROULETTE = "Roulette Wheel Selection"
    SELECTION_TOURNAMENT = "Tournament Selection"
    SELECTION_RANKED = "Ranked Selection"
    SELECTION_FITNESS_SCALING = "Fitness Scaling Selection"
    selectionTypes = [
        SELECTION_ROULETTE,
        SELECTION_TOURNAMENT,
        SELECTION_RANKED,
        SELECTION_FITNESS_SCALING,
    ]

    def __init__(self, scenario, time_allowance=60.0):
        """Initialize the genetic algorithm solver."""
        self._scenario = scenario
        self._timeAllowance = time_allowance
        self._generation = 0
        self._population = []
        self._children = []
        self._bssf = None

        # General parameters
        self.populationSize = 100
        self.newChildrenPerGeneration = 50
        self.maxGenerationsNoChange = 100

        # Crossover parameters
        self.numCrossoversPerGeneration = 50
        self.numCrossoverSplits = 2
        self.crossoverSelectionType = self.SELECTION_ROULETTE

        # Mutation parameters
        self.numMutationsPerGeneration = 50
        self.numMutationsPerSolution = 2
        self.mutationSelectionType = self.SELECTION_ROULETTE

        # Survivor selection parameters
        self.percentOldSurvivors = 0.5
        self.survivorSelectionType = self.SELECTION_ROULETTE

        self.tournamentSize = 5

    def solve(self) -> dict:
        """Solve the genetic algorithm problem."""
        self._startTime = time.time()
        self._generation = 0

        self.initializePopulation()
        self._bssf = self._population[0]

        while True:
            self._generation += 1

            self.crossover()

            self.mutation()

            self.evaluate()

            self.survivorSelection()

            # Check if the time is up
            if time.time() - self._startTime > self._timeAllowance:
                break

            # Check if the best solution has changed in a while
            if self._generation - self._bssf._generation > self.maxGenerationsNoChange:
                break

        solution = TSPSolution(self._bssf._solution)
        endTime = time.time()

        results = {}
        results["cost"] = solution.cost
        results["time"] = endTime - self._startTime
        results["soln"] = solution
        results["count"] = 1
        results["max"] = None
        results["total"] = None
        results["pruned"] = None

        return results

    def initializePopulation(self):
        """Initialize the population for the genetic algorithm."""
        for i in range(self.populationSize):
            self._population.append(self.createRandomSolution(self._generation))

    def createRandomSolution(self, generation):
        """Create a random solution for the genetic algorithm."""
        cities = self._scenario.getCities()
        ncities = len(cities)

        foundTour = False
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])

            solution = GeneticSolution(route, generation)
            if solution.calculateFitness() < np.inf:
                # Found a valid route
                foundTour = True

        return solution

    def crossover(self):  # TODO: Implement
        generation_crossovers_perfromed = 0
        while generation_crossovers_perfromed < self.numCrossoversPerGeneration:
            parents = self.selectParents()
            parent1 = parents[0]._solution
            parent2 = parents[1]._solution

            first_index = randrange(len(parent1))
            second_index = randrange(len(parent1))
            child1 = [math.inf] * len(parent1)
            child2 = [math.inf] * len(parent1)

            if first_index > second_index:
                temp = first_index
                first_index = second_index
                second_index = temp

            child1 = (
                child1[0:first_index]
                + parent1[first_index:second_index]
                + child1[second_index:]
            )
            for i in range(first_index, second_index):
                if parent2[i] not in child1:
                    parent1_temp = parent1[i]
                    while True:
                        parent2_index = parent2.index(parent1_temp)
                        if parent2_index < first_index or parent2_index >= second_index:
                            break
                        else:
                            parent1_temp = parent1[parent2_index]
                    child1[parent2_index] = parent2[i]
            for i in range(len(parent1)):
                child1[i] = child1[i] if child1[i] != math.inf else parent2[i]

            child2 = (
                child2[0:first_index]
                + parent2[first_index:second_index]
                + child2[second_index:]
            )
            for i in range(first_index, second_index):
                if parent1[i] not in child2:
                    parent2_temp = parent2[i]
                    while True:
                        parent1_index = parent1.index(parent2_temp)
                        if parent1_index < first_index or parent1_index >= second_index:
                            break
                        else:
                            parent2_temp = parent2[parent1_index]
                    child2[parent1_index] = parent1[i]
            for i in range(len(parent2)):
                child2[i] = child2[i] if child2[i] != math.inf else parent1[i]

            self._children.append(GeneticSolution(child1, self._generation))
            self._children.append(GeneticSolution(child2, self._generation))
            generation_crossovers_perfromed += 1

    def mutation(self):
        """Perform mutation for the genetic algorithm."""
        generation_mutations_performed = 0
        mutated = []
        while generation_mutations_performed < self.numMutationsPerGeneration:
            to_mutate = self._population[randrange(self.populationSize)]
            if self.mutationSelectionType == self.SELECTION_TOURNAMENT:
                to_mutate = self.tournamentSelection(
                    self._population, self.tournamentSize
                )
            elif self.mutationSelectionType == self.SELECTION_ROULETTE:
                to_mutate = self.rouletteSelection(self._population)
            elif self.mutationSelectionType == self.SELECTION_RANKED:
                to_mutate = self.rankedSelection(self._population)
            elif self.mutationSelectionType == self.SELECTION_FITNESS_SCALING:
                to_mutate = self.fitnessScalingSelection(self._population)

            # if to_mutate in mutated:
            #     continue
            mutated.append(to_mutate)
            generation_mutations_performed += 1
            old_route = copy(to_mutate._solution)
            route_mutations_performed = 0
            while route_mutations_performed < self.numMutationsPerSolution:
                first_index = randrange(len(old_route))
                second_index = randrange(len(old_route))
                temp = old_route[first_index]
                old_route[first_index] = old_route[second_index]
                old_route[second_index] = temp
                route_mutations_performed += 1
            solution = GeneticSolution(old_route, self._generation)
            self._children.append(solution)

    def evaluate(self):
        """Evaluate the population for the genetic algorithm."""
        for solution in self._population:
            solution.calculateFitness()
            if solution._fitness < self._bssf._fitness:
                self._bssf = solution
        # Do we need this once we have actual selection?
        for solution in self._children:
            solution.calculateFitness()
            if solution._fitness < self._bssf._fitness:
                self._bssf = solution

    def survivorSelection(self):  # TODO: Implement
        """Perform survivor selection for the genetic algorithm."""
        num_old_survivors = int(self.percentOldSurvivors * self.populationSize)
        num_new_survivors = self.populationSize - num_old_survivors

        selected = []
        num_old_selected = 0
        num_new_selected = 0
        while num_old_selected < num_old_survivors:
            if self.survivorSelectionType == self.SELECTION_TOURNAMENT:
                selected.append(self.tournamentSelection(self._population, 2))
            elif self.survivorSelectionType == self.SELECTION_ROULETTE:
                selected.append(self.rouletteSelection(self._population))
            elif self.survivorSelectionType == self.SELECTION_RANKED:
                selected.append(self.rankedSelection(self._population))
            elif self.survivorSelectionType == self.SELECTION_FITNESS_SCALING:
                selected.append(self.fitnessScalingSelection(self._population))
            num_old_selected += 1
        while num_new_selected < num_new_survivors:
            if self.survivorSelectionType == self.SELECTION_TOURNAMENT:
                selected.append(self.tournamentSelection(self._children, 2))
            elif self.survivorSelectionType == self.SELECTION_ROULETTE:
                selected.append(self.rouletteSelection(self._children))
            elif self.survivorSelectionType == self.SELECTION_RANKED:
                selected.append(self.rankedSelection(self._children))
            elif self.survivorSelectionType == self.SELECTION_FITNESS_SCALING:
                selected.append(self.fitnessScalingSelection(self._children))
            num_new_selected += 1
        temp_set = set(self._population)
        print("Num unqique cities = ", len(temp_set))
        self._population = selected
        temp_set = set(selected)
        print("Num unqique cities = ", len(temp_set))
        self._children = []
        # TEMP unitl seleciton functions work
        # self._population = self._population[0:50] + self._children[0:50]

    def selectParents(self):
        """Select parents for the genetic algorithm."""
        parents = []
        while len(parents) < 2:
            if self.crossoverSelectionType == self.SELECTION_ROULETTE:
                parent = self.rouletteSelection(self._population)
            elif self.crossoverSelectionType == self.SELECTION_TOURNAMENT:
                parent = self.tournamentSelection(self._population)
            elif self.crossoverSelectionType == self.SELECTION_RANKED:
                parent = self.rankedSelection(self._population)
            elif self.crossoverSelectionType == self.SELECTION_FITNESS_SCALING:
                parent = self.fitnessScalingSelection(self._population)
            if parent not in parents:
                parents.append(parent)
        return parents

    def rouletteSelection(self, population):
        """Perform roulette selection for the genetic algorithm."""
        # fitness represents city distance, so use inverse so lower fitnesses are more
        # likely to be chosen
        fitnessValues = [((1 / city.calculateFitness()) * 1000) for city in population]
        totalFitness = int(sum(fitnessValues))
        rand = random.randint(0, totalFitness)

        partialSum = 0
        for city in population:
            partialSum += 1 / city.calculateFitness() * 1000
            if partialSum >= rand:
                return city

    def tournamentSelection(self, population, tournamentSize=5):
        """Perform tournament selection for the genetic algorithm."""
        """Note that the tournamentSize parameter sets the number 
        of solutions that will compete in each tournament. The higher 
        the value of tournamentSize, the stronger the selection pressure will be, 
        and the more likely it is that the best solutions will be selected as parents. 
        However, larger tournaments also increase the risk of premature convergence, 
        because they reduce the diversity of the population. Therefore, you should 
        experiment with different values of tournamentSize to find the one that works 
        best for your problem."""
        # Note: if using this for survival selection, you need to call this function population size times?
        # could consider lowering chance of duplications by removing selected cities from populations?
        participants = []
        while len(participants) <= tournamentSize:
            selectedCity = population[random.randint(0, len(population) - 1)]
            if selectedCity not in participants:
                participants.append(selectedCity)
        return min(participants, key=lambda x: x._fitness)

    def rankedSelection(self, population):
        """Perform ranked selection for the genetic algorithm."""
        # Sort the solutions by their fitness scores
        population = sorted(population, key=lambda city: city.calculateFitness())

        # Assign ranks to each solution
        ranks = {}
        for i, city in enumerate(population):
            ranks[city] = len(population) - i

        # Calculate sum of ranks
        totalRank = sum([ranks[city] for city in population])

        # Make your selection based on rank order
        rand = random.randint(0, totalRank)
        partialSum = 0
        for city in population:
            partialSum += ranks[city]
            if partialSum >= rand:
                return city

    def fitnessScalingSelection(
        self, population
    ):  # TODO: fix bug causing None to be selected
        # consider making this a check box rather than one of the drop
        # down options since it can be used with a selection
        """Perform fitness scaling selection for the genetic algorithm."""
        fitnessValues = [city.calculateFitness() for city in population]
        minFitness = min(fitnessValues)
        maxFitness = max(fitnessValues)

        # Scale values to fit in the 0 - 100 range
        scaledValues = {}
        totalFitness = 0
        for city in population:
            scaledVal = (city.calculateFitness() - minFitness) * (
                100 / (maxFitness - minFitness)
            )
            scaledValues[city] = scaledVal
            totalFitness += scaledVal

        # Begin selection on newly scaled fitness values
        fitnessValues = []
        for city in population:
            if scaledValues[city] != 0:
                fitnessValues.append((1 / scaledValues[city]) * 1000)
            else:
                fitnessValues.append(0)

        # Make selection
        rand = random.randint(0, totalFitness)
        partialSum = 0
        for city in population:
            if scaledValues[city] != 0:
                partialSum += 1 / scaledValues[city] * 1000
            if partialSum >= rand:
                return city


class GeneticSolution:
    """Solution for the genetic algorithm.

    Attributes:
        _solution (list): The list of cities in the solution.
        _fitness (int): The fitness of the solution.
        _generation (int): The generation the solution was created in.
    """

    def __init__(self, route: list, generation: int) -> None:
        """Initialize the genetic solution."""
        self._solution = route
        self._fitness = self.calculateFitness()
        self._generation = generation

    def calculateFitness(self) -> int:
        """Calculate the fitness of the solution."""
        totalFitness = 0
        for i, city in enumerate(self._solution):
            nextCity = self._solution[(i + 1) % len(self._solution)]
            totalFitness += city.costTo(nextCity)

        self._fitness = totalFitness
        return totalFitness


# GREEDY ALGORITHM
def greedyTSP(cities, time_allowance=60.0, startIndex=0, startTime=None):
    """Compute a solution to the TSP problem for the scenario using a greedy algorithm.

    The time complexity of this algorithm is O(n^2) because it has to find the closest city to the current city for
    every city in the list.

    The space complexity of this algorithm is O(n) because it has to store the route and the list of cities to search.

    Arguments:
        cities (list): The list of cities to find a route for.
        time_allowance (float): The amount of time to allow the algorithm to run for. Defaults to 60 seconds.
        startIndex (int): The index of the city to start the route with. Defaults to 0.
        startTime (float): The time the algorithm started running. Defaults to None, in which case the current time is
            used.
    """
    results = {}
    if startTime is None:
        startTime = time.time()

    startCity = cities[startIndex]
    route = [startCity]
    citiesToSearch = cities.copy()
    citiesToSearch.remove(startCity)
    totalCost = 0

    while len(citiesToSearch) > 0:
        closestCity = None
        closestDistance = math.inf
        for city in citiesToSearch:
            distance = startCity.costTo(city)
            if distance < closestDistance:
                closestDistance = distance
                closestCity = city

        if closestDistance == math.inf:
            break

        midTime = time.time()
        if midTime - startTime > time_allowance:
            results = {}
            results["cost"] = math.inf
            results["time"] = midTime - startTime
            results["soln"] = None
            results["count"] = len(route)
            results["max"] = None
            results["total"] = None
            results["pruned"] = None
            return results

        route.append(closestCity)
        citiesToSearch.remove(closestCity)
        startCity = closestCity
        totalCost += closestDistance

    # If the route found isn't complete, run the alrogithm again with a different starting city
    if len(route) != len(cities) or route[-1].costTo(route[0]) == math.inf:
        print("Route not complete, running again with different starting city")
        return greedyTSP(
            cities,
            time_allowance=time_allowance,
            startIndex=startIndex + 1,
            startTime=startTime,
        )

    solution = TSPSolution(route)
    endTime = time.time()

    results["cost"] = solution.cost
    results["time"] = endTime - startTime
    results["soln"] = solution
    results["count"] = 1
    results["max"] = None
    results["total"] = None
    results["pruned"] = None

    return results


# BRANCH AND BOUND ALGORITHM
class Node:
    """Class to hold information for each node in the branch and bound tree.

    Attributes:
        state (list): The current state of the node, represented by a cost matrix.
        score (int): The score of the node, which is the lower bound of the node.
        path (list): The path of the node, represented by a list of City objects.
    """

    def __init__(self, city, scenario=None, parentNode=None) -> None:
        """Initialize the Node object.

        Time complexity: O(n^2) because we need to create a cost matrix for the node and populate it with the costs.
        Space complexity: O(n^2) because we need to create a cost matrix for the node.

        Arguments:
            city (City): The city to add to the parent node's path to initialize the node.
            scenario (Scenario): The scenario to use to initialize the node.
            parentNode (Node): The parent node to use to initialize the node.
        """
        if scenario is not None:
            numCities = len(scenario.getCities())
        elif parentNode is not None:
            numCities = parentNode.state.shape[0]

        self.state = np.empty((numCities, numCities), dtype=float)
        self.score = 0
        self.path = []
        self.depth = 0

        if scenario is not None:
            self.setupStateScenario(scenario, city)
        elif parentNode is not None:
            self.setupStateParent(parentNode, city)

    def setupStateScenario(self, scenario, city):
        """Set up the state matrix for a node using a scenario.

        Time complexity: O(n^2) because we need to create a cost matrix for the node and populate it with the costs.
        Space complexity: O(n^2) because we need to create a cost matrix for the node.
        """
        # Set up the state matrix
        for i, city1 in enumerate(scenario.getCities()):
            for j, city2 in enumerate(scenario.getCities()):
                if i == j:
                    self.state[i, j] = np.inf
                else:
                    self.state[i, j] = city1.costTo(city2)

        self.path = [city]
        # Reduce the state matrix
        self.reduceCostMatrix()

    def setupStateParent(self, parentNode, city):
        """Set up the state matrix for a node using a parent node.

        Time complexity: O(n^2) because we need to create a cost matrix for the node and populate it with the costs.
        Space complexity: O(n^2) because we need to create a cost matrix for the node.
        """
        self.state = parentNode.state.copy()
        self.score = parentNode.score
        self.depth = parentNode.depth + 1

        # Set up the path
        self.path = parentNode.path.copy()
        previousCity = self.path[-1]
        self.path.append(city)

        # Add the cost of the path to the score
        self.score += self.state[previousCity._index][city._index]

        # Set everything in the "from row" to infinity
        self.state[previousCity._index, :] = np.inf

        # Set everything in the "to column" to infinity
        self.state[:, city._index] = np.inf

        # Set the reverse path to infinity
        self.state[city._index, previousCity._index] = np.inf

        # Reduce the state matrix
        self.reduceCostMatrix()

    def reduceCostMatrix(self):
        """Run the reduce cost matrix algorithm on the node's state matrix and update self.score.

        Time complexity: O(n^2) because we need to iterate over every element in the state matrix.
        Space complexity: O(1) because we don't need to create any new data structures.

            Any update to the state matrix will also have a time complexity of O(n^2) and a space complexity of O(1),
        since we just need to rerun this function.
        """
        # Iterate over rows
        for i in range(self.state.shape[0]):
            if np.all(np.isinf(self.state[i, :])):
                continue

            # Find the minimum value in the row
            minVal = np.min(self.state[i, :])
            if minVal == np.inf or minVal == 0:
                continue

            # Subtract the minimum value from every element in the row
            self.state[i, :] -= minVal
            self.score += minVal

        # Iterate over columns
        for i in range(self.state.shape[1]):
            if np.all(np.isinf(self.state[:, i])):
                continue
            # Find the minimum value in the column
            minVal = np.min(self.state[:, i])
            if minVal == np.inf or minVal == 0:
                continue
            # Subtract the minimum value from every element in the column
            self.state[:, i] -= minVal
            self.score += minVal

    def __lt__(self, other):
        """Override the less than operator to compare nodes by depth.

        Time complexity: O(1) because we are just comparing two integers.
        Space complexity: O(1) because we are not creating any new data structures.
        """
        return self.depth < other.depth

    def __gt__(self, other):
        """Override the greater than operator to compare nodes by depth.

        Time complexity: O(1) because we are just comparing two integers.
        Space complexity: O(1) because we are not creating any new data structures.
        """
        return self.depth > other.depth

    def isSolution(self) -> bool:
        """Return if the current node is a solution by looking at the number of nodes in the path.

        Time complexity: O(1) because we are just comparing two integers.
        Space complexity: O(1) because we are not creating any new data structures.
        """
        return len(self.path) == self.state.shape[0]


class BranchAndBound:
    """Class to hold the branch and bound algorithm.

    Attributes:
        _scenario (Scenario): The scenario to solve.
        _bssf (TSPSolution): The best solution found so far.
        _startTime (float): The time the algorithm started.
        _timeAllowance (float): The time allowance for the algorithm.
        _numSolutions (int): The number of solutions found.
        _numPruned (int): The number of nodes pruned.
        _numMax (int): The number of nodes in the max heap.
        _numTotal (int): The number of nodes in the total heap.
    """

    def __init__(self, scenario: Scenario, time_allowance: float) -> None:
        """Initialize the BranchAndBound object.

        Time complexity: O(1) because we are just setting the attributes.
        Space complexity: O(1) because we are not creating any new data structures.
        """
        self._scenario = scenario
        self._timeAllowance = time_allowance
        self._bssf = np.inf
        self._bssfPath = []
        self._startTime = -1
        self._numNodes = 0
        self._numPruned = 0
        self._numSolutions = 0
        self.priorityQueue = []

        self.minBSSFBeforeSwitch = len(self._scenario.getCities()) // 4

    def solve(self) -> dict:
        """Solve the TSP problem using branch and bound.

        Time complexity: In the absolute worst case, the time complexity of this algorithm is O(n!) because we need to
        iterate over every possible solution. However, in practice, the time complexity is much better because we
        prune nodes that are not promising. This makes our time complexity in practice something more along the lines
        of O(n^2 * 2^n) because we need to iterate over every node in the tree and we need to iterate over every
        possible solution.

        Space complexity: O(n^4) because we can assume that for each of the n cities, we need to create another n
        children, and each child needs to create a copy of the state matrix, which is n x n. This gives us a total
        space complexity of O(n^4).

        Returns:
            results dictionary for GUI that contains three ints: cost of best solution,
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this
            algorithm
        """
        self._startTime = time.time()

        # INITIAL BSSF
        # Find the initial BSSF using the greedy algorithm
        # The time complexity of this algorithm is O(n^2) because we need to iterate over every city
        # The space complexity of this algorithm is O(n) because we need to create a copy of the cities
        cities = self._scenario.getCities().copy()
        greedyResults = greedyTSP(cities=cities, time_allowance=self._timeAllowance)

        self._bssf = greedyResults["cost"]
        solution = TSPSolution(greedyResults["soln"].route)
        self._bssfPath = solution.route
        self._numSolutions += 1
        print(f"Greedy solution: {self._bssf}")

        self.searchByDepth = True
        print(
            f"Switching to branch and bound after {self.minBSSFBeforeSwitch} solutions"
        )

        # Create the initial node
        startCity = self._scenario.getCities()[0]
        node = Node(scenario=self._scenario, city=startCity)
        self._numNodes += 1

        # PRIORITY QUEUE
        # The priority queue is implemented using a binary heap, so the time complexity of adding and removing
        # elements is O(log(n)).
        # The space complexity would be, at worst, O((n-1)!) because we would need to store every possible
        # permutation of the cities in the priority queue. However, we can prune a lot of these permutations
        # by using the lower bound, which improves the space complexity quite a bit.
        heapq.heappush(self.priorityQueue, (node.score, node))

        maxQueueSize = len(self.priorityQueue)

        while len(self.priorityQueue) > 0:
            # Get the next node
            node = heapq.heappop(self.priorityQueue)[1]

            if node.score >= self._bssf:
                self._numPruned += 1
                continue

            # Check if we're at a solution
            if node.isSolution():
                # Check if the solution is better than the current best
                if node.score < self._bssf:
                    self._bssf = node.score
                    self._bssfPath = node.path
                    self._numSolutions += 1
                    print(
                        f"Found a better solution! (#{self._numSolutions}: {self._bssf})"
                    )

                    if (
                        self._numSolutions > self.minBSSFBeforeSwitch
                        and self.searchByDepth
                    ):
                        self.searchByDepth = False
                        print(
                            "Switching to prioritizing lower bound over depth. "
                            f"Heapifying {len(self.priorityQueue)} nodes...)"
                        )
                        # Redo the heap with the new priority
                        self.priorityQueue = [
                            (node.score, node) for score, node in self.priorityQueue
                        ]
                        heapq.heapify(self.priorityQueue)
                else:
                    self._numPruned += 1
            else:
                # Expand the node
                newNodes = self.expandNode(node)
                self._numNodes += len(newNodes)
                for newNode in newNodes:
                    if self._numSolutions > self.minBSSFBeforeSwitch:
                        # Prioritize lower bound over depth
                        heapq.heappush(self.priorityQueue, (newNode.score, newNode))
                    else:
                        # Prioritize depth over lower bound
                        heapq.heappush(
                            self.priorityQueue, (newNode.depth * -1, newNode)
                        )

            # Update the max queue size
            if len(self.priorityQueue) > maxQueueSize:
                maxQueueSize = len(self.priorityQueue)

            # Check if the time is up
            if time.time() - self._startTime > self._timeAllowance:
                print("Time's up!")
                for score, node in self.priorityQueue:
                    if score > self._bssf:
                        self._numPruned += 1
                break

        solution = TSPSolution(self._bssfPath)

        endTime = time.time()
        results = {}
        results["cost"] = solution.cost
        results["time"] = endTime - self._startTime
        results["soln"] = solution
        results["count"] = self._numSolutions
        results["max"] = maxQueueSize
        results["total"] = self._numNodes
        results["pruned"] = self._numPruned

        return results

    def expandNode(self, node) -> list:
        """Expand the given node.

        Time complexity: O(n^3) because we are creating new Nodes for at most n-1 cities. That opteration takes O(n^2)
            time because we are reducing the cost matrix for each node, so overall, our time complexity is O(n^3).
        Space complexity: O(n^3) because we need to create a new node for every city that is not already in the path.
            Each node has a cost matrix that is O(n^2) in size, so our space complexity is O(n^3).

        Arguments:
            node (Node): The node to expand.

        Returns:
            A tuple containing a list of new nodes and the number of nodes pruned.
        """
        newNodes = []
        numPruned = 0

        for city in self._scenario.getCities():
            # Don't consider already visited nodes
            if city in node.path:
                continue

            # Don't consider nodes that are impossible to reach
            if node.state[node.path[-1]._index, city._index] == np.inf:
                numPruned += 1
                continue

            # Create a new node
            newNode = Node(parentNode=node, city=city)
            newNode.reduceCostMatrix()
            newNodes.append(newNode)

        self._numPruned += numPruned

        return newNodes
