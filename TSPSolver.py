#!/usr/bin/python3

from copy import copy
import heapq
import itertools
from random import randrange
import time
import logging

import numpy as np
from PyQt6.QtCore import QLineF, QPointF
from TSPClasses import *


logging.basicConfig(filename="TSP.log", level=logging.DEBUG, format="%(message)s")


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

    # Initialization types
    INITIALIZATION_PURE_RANDOM = "Pure Random Initialization"
    INITIALIZATION_RANDOM = "Random Valid Initialization"
    INITIALIZATION_IMPROVED_RANDOM = "Improved Random Initialization"
    INITIALIZATION_GREEDY = "Greedy Initialization"
    initializationTypes = [
        INITIALIZATION_PURE_RANDOM,
        INITIALIZATION_RANDOM,
        INITIALIZATION_IMPROVED_RANDOM,
        INITIALIZATION_GREEDY,
    ]

    def __init__(self, scenario, time_allowance=60.0):
        """Initialize the genetic algorithm solver."""
        self._scenario = scenario
        self._timeAllowance = time_allowance
        self._generation = 0
        self._population = []
        self._children = []
        self._bssf = None
        self.selectableOptionsMap = {}

        # General parameters
        self.populationSize = 100
        self.maxGenerationsNoChange = 100
        self.pruneInfinites = False
        self.initializationType = self.INITIALIZATION_GREEDY

        # Crossover parameters
        self.numCrossoversPerGeneration = 50
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
        self._population = []
        self._children = []

        startTime = time.time()
        self.initializePopulation()
        endTime = time.time()
        print(f"Population initialization took {endTime - startTime} seconds.")
        self._bssf = self._population[0]
        self.bssf_updates = 0

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

        print(
            f"Found solution {solution.cost} after {self._generation} generations in {endTime - self._startTime} seconds."
        )

        results = {}
        results["cost"] = solution.cost
        results["time"] = endTime - self._startTime
        results["soln"] = solution
        results["count"] = self.bssf_updates
        results["max"] = None
        results["total"] = self._generation
        results["pruned"] = None

        # LOGGING
        logging.debug("PARAMETERS")
        logging.debug(f"numCities: {len(self._scenario.getCities())}")
        logging.debug(f"populationSize: {self.populationSize}")
        logging.debug(f"maxGenerationsNoChange: {self.maxGenerationsNoChange}")
        logging.debug(f"pruneInfinites: {self.pruneInfinites}")
        logging.debug(f"numCrossoversPerGeneration: {self.numCrossoversPerGeneration}")
        logging.debug(f"crossoverSelectionType: {self.crossoverSelectionType}")
        logging.debug(f"numMutationsPerGeneration: {self.numMutationsPerGeneration}")
        logging.debug(f"numMutationsPerSolution: {self.numMutationsPerSolution}")
        logging.debug(f"mutationSelectionType: {self.mutationSelectionType}")
        logging.debug(f"percentOldSurvivors: {self.percentOldSurvivors}")
        logging.debug(f"survivorSelectionType: {self.survivorSelectionType}")
        logging.debug(f"tournamentSize: {self.tournamentSize}")

        logging.debug("\nRESULTS")
        logging.debug(f"cost: {results['cost']}")
        logging.debug(f"time: {results['time']}")
        logging.debug(f"generation: {self._generation}")
        logging.debug(f"bssf_updates: {results['count']}")

        logging.debug("\n\n")

        return results

    def initializePopulation(self):
        """Initialize the population for the genetic algorithm."""
        initializationFunction = None
        if self.initializationType == self.INITIALIZATION_PURE_RANDOM:
            initializationFunction = self.createPureRandomSolution
        elif self.initializationType == self.INITIALIZATION_RANDOM:
            initializationFunction = self.createRandomSolution
        elif self.initializationType == self.INITIALIZATION_IMPROVED_RANDOM:
            initializationFunction = self.createRandomSolutionBetter
        elif self.initializationType == self.INITIALIZATION_GREEDY:
            initializationFunction = self.createRandomSolutionGreedy
        else:
            raise Exception(f"Invalid initialization type {self.initializationType}")

        for i in range(self.populationSize):
            self._population.append(initializationFunction())

    def createRandomSolution(self):
        """Create a random solution for the genetic algorithm."""
        cities = self._scenario.getCities()
        ncities = len(cities)

        foundTour = False
        i = 0
        while not foundTour:
            # create a random permutation
            i += 1
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])

            solution = GeneticSolution(route, self._generation)
            if solution.calculateFitness() < np.inf:
                # Found a valid route
                foundTour = True
                print(f"Found a valid route after {i} attempts")
        return solution

    def createPureRandomSolution(self):
        """Create a pure random solution without worrying if it's valid."""
        cities = self._scenario.getCities()
        ncities = len(cities)

        # create a random permutation
        perm = np.random.permutation(ncities)
        route = []
        # Now build the route using the random permutation
        for i in range(ncities):
            route.append(cities[perm[i]])

        solution = GeneticSolution(route, self._generation)
        return solution

    def createRandomSolutionBetter(self):
        """Create a random solution that's hopefully better than the previous implementation."""
        cities = self._scenario.getCities()
        ncities = len(cities)

        foundTour = False
        while not foundTour:
            # Pick a starting city
            startCity = cities[randrange(ncities)]

            # Create a list of cities to visit
            citiesToVisit = cities.copy()
            citiesToVisit.remove(startCity)

            # Create a route
            route = [startCity]
            i = 0
            while len(citiesToVisit) > 0:
                # Pick a random city and see if there is a path to it
                randomCity = citiesToVisit[randrange(len(citiesToVisit))]
                i += 1
                if i > 50:
                    # We've tried 50 times to find a path to a random city, so start over
                    break
                # if self._scenario.getDistance(route[-1], randomCity) < np.inf:
                if route[-1].costTo(randomCity) < np.inf:
                    # There is a path to the random city, so add it to the route
                    route.append(randomCity)
                    citiesToVisit.remove(randomCity)
                    i = 0

            if len(citiesToVisit) == 0:
                # We found a valid route
                foundTour = True
        return GeneticSolution(route, self._generation)

    def createRandomSolutionGreedy(self):
        """Create a random solution for the genetic algorithm."""
        randomStartIndex = randrange(len(self._scenario.getCities()))
        solution = greedyTSP(
            self._scenario.getCities(),
            time_allowance=self._timeAllowance,
            startIndex=randomStartIndex,
        )
        route = solution["soln"].route
        return GeneticSolution(route, self._generation)

    # O(num_crossovers * n)
    def crossover(self):
        generation_crossovers_perfromed = 0
        interations = 0
        # Do as many crossovers user picked in the GUI
        # O(num_cross_overs)
        while generation_crossovers_perfromed < self.numCrossoversPerGeneration:
            interations += 1
            if interations >= 10000:
                raise Exception(
                    f"Stuck in the crossover function for more than 10000 iterations"
                )
            # O(selction functions)
            parents = self.selectParents()
            parent1 = parents[0]._solution
            parent2 = parents[1]._solution

            # Get a starting and ending index and initialize the children to inf
            # O(1)
            first_index = randrange(len(parent1))
            second_index = randrange(len(parent2))
            child1 = [math.inf] * len(parent1)
            child2 = [math.inf] * len(parent2)

            # Make sure our indexes are in order
            # O(1)
            if first_index > second_index:
                temp = first_index
                first_index = second_index
                second_index = temp

            # Copy the swath from parent one into the child
            child1 = (
                child1[0:first_index]
                + parent1[first_index:second_index]
                + child1[second_index:]
            )
            # Fill in the child with parts from parent 2
            # O(n)
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
            # Fill in anything not yet filled out
            for i in range(len(parent1)):
                child1[i] = child1[i] if child1[i] != math.inf else parent2[i]
            # O(n)
            child1 = GeneticSolution(child1, self._generation)
            # Don't add if we are pruning infinities and it is an infinite cost
            if self.pruneInfinites and child1._fitness == math.inf:
                continue
            # Same thing for child2, but with the parents swithced
            # O(n)
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
            # O(n)
            child2 = GeneticSolution(child2, self._generation)
            if self.pruneInfinites and child2._fitness == math.inf:
                continue
            # Add both children and increment the counter
            self._children.append(child1)
            self._children.append(child2)
            generation_crossovers_perfromed += 1

    # O(num_mutations * n)
    def mutation(self):
        """Perform mutation for the genetic algorithm."""
        generation_mutations_performed = 0
        iterations = 0
        # Perform as many as the gui asked
        # O(num_mutations)
        while generation_mutations_performed < self.numMutationsPerGeneration:
            iterations += 1
            # Select the parent
            to_mutate = self._population[randrange(self.populationSize)]
            # O(selection_functions)
            if self.mutationSelectionType == self.SELECTION_TOURNAMENT:
                to_mutate = self.tournamentSelection(
                    self._population, self.tournamentSize
                )
            elif self.mutationSelectionType == self.SELECTION_ROULETTE:
                to_mutate = self.rouletteSelection(self._population, [])
            elif self.mutationSelectionType == self.SELECTION_RANKED:
                to_mutate = self.rankedSelection(self._population)
            elif self.mutationSelectionType == self.SELECTION_FITNESS_SCALING:
                to_mutate = self.fitnessScalingSelection(self._population, [])
            # O(n)
            old_route = copy(to_mutate._solution)
            route_mutations_performed = 0
            # Swap two random indexes
            # O(num_muations) = O(1)
            while route_mutations_performed < self.numMutationsPerSolution:
                first_index = randrange(len(old_route))
                second_index = randrange(len(old_route))
                temp = old_route[first_index]
                old_route[first_index] = old_route[second_index]
                old_route[second_index] = temp
                route_mutations_performed += 1
            # Create the soultion and check if it is infinity
            # O(n)
            solution = GeneticSolution(old_route, self._generation)
            if self.pruneInfinites and solution._fitness == math.inf:
                if iterations >= 10000:
                    raise Exception(
                        f"Stuck in mutation function for more thatn 10000 iterations"
                    )
                continue
            self._children.append(solution)
            generation_mutations_performed += 1

    def evaluate(self):
        """Evaluate the population for the genetic algorithm."""
        for solution in self._population:
            if solution._fitness < self._bssf._fitness:
                self._bssf = solution
                self.bssf_updates += 1
        # Do we need this once we have actual selection?
        for solution in self._children:
            if solution._fitness < self._bssf._fitness:
                self._bssf = solution
                self.bssf_updates += 1

    # O(pop_size * selection_functions)
    def survivorSelection(self):
        """Perform survivor selection for the genetic algorithm."""
        num_old_survivors = int(self.percentOldSurvivors * self.populationSize)
        num_new_survivors = self.populationSize - num_old_survivors
        selected = set()
        iterations = 0
        while len(selected) < num_old_survivors:
            if self.survivorSelectionType == self.SELECTION_TOURNAMENT:
                selected.add(
                    self.tournamentSelection(self._population, self.tournamentSize)
                )
            elif self.survivorSelectionType == self.SELECTION_ROULETTE:
                selected.add(self.rouletteSelection(self._population, selected))
            elif self.survivorSelectionType == self.SELECTION_RANKED:
                selected.add(self.rankedSelection(self._population))
            elif self.survivorSelectionType == self.SELECTION_FITNESS_SCALING:
                selected.add(self.fitnessScalingSelection(self._population, selected))
            iterations += 1
            if iterations >= 10000:
                raise Exception(
                    f"stuck in survivor selection for the old gen for more than 10000 iterations"
                )
        iterations = 0
        while len(selected) < num_new_survivors + num_old_survivors:
            if self.survivorSelectionType == self.SELECTION_TOURNAMENT:
                selected.add(
                    self.tournamentSelection(self._children, self.tournamentSize)
                )
            elif self.survivorSelectionType == self.SELECTION_ROULETTE:
                selected.add(self.rouletteSelection(self._children, selected))
            elif self.survivorSelectionType == self.SELECTION_RANKED:
                selected.add(self.rankedSelection(self._children))
            elif self.survivorSelectionType == self.SELECTION_FITNESS_SCALING:
                selected.add(self.fitnessScalingSelection(self._children, selected))
            iterations += 1
            if iterations >= 10000:
                raise Exception(
                    f"stuck in survivor selection for the new gen for more than 10000 iterations"
                )
        self._population = list(selected)
        self._children = []

    # O(selection_functions)
    def selectParents(self):
        """Select parents for the genetic algorithm."""
        parents = []
        while len(parents) < 2:
            if self.crossoverSelectionType == self.SELECTION_ROULETTE:
                parent = self.rouletteSelection(self._population, parents)
            elif self.crossoverSelectionType == self.SELECTION_TOURNAMENT:
                parent = self.tournamentSelection(self._population, self.tournamentSize)
            elif self.crossoverSelectionType == self.SELECTION_RANKED:
                parent = self.rankedSelection(self._population)
            elif self.crossoverSelectionType == self.SELECTION_FITNESS_SCALING:
                parent = self.fitnessScalingSelection(self._population, parents)
            if parent not in parents:
                parents.append(parent)
        return parents

    def printInfo(self, population):
        temp_set = set(population)
        print("Num unqique routes = ", len(temp_set))
        temp_set.clear()
        num_sol = 0
        for p in population:
            if p._fitness != np.inf:
                num_sol += 1
                temp_set.add(p)
        print("Num of solutions", num_sol)
        print("Num of unique valid solutions", len(temp_set))

    def rouletteSelection(self, population, selected):
        """Perform roulette selection for the genetic algorithm."""
        # Create an array of inverted (and scaled) fitness values (so smaller values win)
        fitnessValues = [
            ((1 / city._fitness)) for city in population if city not in selected
        ]

        # Total our new massage fitness values
        totalFitness = sum(fitnessValues)

        # Pick a random number in the range of our fitness sum
        rand = random.uniform(0, 1) * totalFitness

        partialSum = 0.0
        for city in population:
            if city not in selected:
                partialSum += 1 / city._fitness
                if partialSum >= rand:
                    return city

    def tournamentSelection(self, population, tournamentSize=5):
        """Perform tournament selection for the genetic algorithm."""
        # Initial array of candidates for selection
        participants = []

        for i in range(tournamentSize):
            selectedCity = population[random.randint(0, len(population) - 1)]
            participants.append(selectedCity)

        # Pick the minimum value among candidates for selection
        selection = min(participants, key=lambda x: x._fitness)

        return selection

    def rankedSelection(self, population):
        """Perform ranked selection for the genetic algorithm."""
        # Sort the solutions by their fitness scores
        population.sort(key=lambda city: city._fitness)

        # Assign ranks to each solution
        ranks = {}
        for i, city in enumerate(population):
            ranks[city] = len(population) - i

        # Calculate selection probabilities
        totalRank = sum([ranks[city] for city in population])

        # Pick a random number in the range of our rank sum
        rand = random.randint(0, totalRank)

        partialSum = 0
        for city in population:
            partialSum += ranks[city]
            if partialSum >= rand:
                return city

    def fitnessScalingSelection(self, population, selected):
        """Perform fitness scaling selection for the genetic algorithm."""
        fitnessValues = [city._fitness for city in population if city not in selected]
        minFitness = min(fitnessValues)
        maxFitness = max(fitnessValues)

        if maxFitness == math.inf:
            maxFitness = 10000000

        if minFitness == math.inf:
            minFitness = 10000000

        # Scale values to fit in the 0 - 100 range
        scaledValues = {}

        for city in population:
            if city not in selected:
                if maxFitness != minFitness:
                    if city._fitness == math.inf:
                        scaledVal = 10000000
                    else:
                        scaledVal = (city._fitness - minFitness) * (
                            100 / (maxFitness - minFitness)
                        )
                else:
                    scaledVal = 0

                scaledValues[city] = scaledVal

        # Roulette selection among the scale values, just because...

        # Create an array of inverted (and scaled) fitness values (so smaller values win)
        fitnessValues = []
        for city in population:
            if city not in selected:
                if scaledValues[city] != 0:
                    fitnessValues.append((1 / scaledValues[city]) * 1000)
                else:
                    fitnessValues.append(0)

        totalFitness = sum(fitnessValues)

        # Pick a random number in the range of our fitness sum
        rand = random.randint(0, int(totalFitness))

        partialSum = 0
        for city in population:
            if city not in selected:
                if scaledValues[city] != 0:
                    partialSum += 1 / scaledValues[city] * 1000

                if partialSum >= rand or len(population) == 1:
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
        # print("Route not complete, running again with different starting city")
        return greedyTSP(
            cities,
            time_allowance=time_allowance,
            startIndex=(startIndex + 1) % len(cities),
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
