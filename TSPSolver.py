#!/usr/bin/python3

import heapq
import itertools
import time

import numpy as np
from PyQt6.QtCore import QLineF, QPointF
from TSPClasses import *


class TSPSolver:
    """TSP Solver."""

    def __init__(self, gui_view):
        """Initialize the TSPSolver."""
        self._scenario = None

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
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
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
        geneticSolver = GeneticSolver(scenario=self._scenario, time_allowance=time_allowance)


# GENETIC ALGORITHM
class GeneticSolver:
    """Genetic algorithm solver for the traveling salesperson problem."""

    # Selection types
    SELECTION_ROULETTE = 0
    SELECTION_TOURNAMENT = 1
    SELECTION_RANKED = 2
    SELECTION_FITNESS_SCALING = 3

    def __init__(self, scenario, time_allowance=60.0):
        """Initialize the genetic algorithm solver."""
        self._scenario = scenario
        self._timeAllowance = time_allowance  # TODO: Do we care about time allowance for genetic algorithm?
        self._generation = 0
        self._population = []

        # General parameters
        self.populationSize = 100
        self.newChildrenPerGeneration = 50

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

    def solve(self):
        """Solve the genetic algorithm problem."""
        while True:
            self._generation += 1

            self.parentSelection()

            self.recombination()

            self.mutation()

            self.evaluate()

            self.survivorSelection()

    def parentSelection(self):
        """Perform parent selection for the genetic algorithm."""
        pass

    def recombination(self):
        """Perform recombination for the genetic algorithm."""
        pass

    def mutation(self):
        """Perform mutation for the genetic algorithm."""
        pass

    def evaluate(self):
        """Evaluate the population for the genetic algorithm."""
        pass

    def survivorSelection(self):
        """Perform survivor selection for the genetic algorithm."""
        pass


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
            results['max'] = None
            results['total'] = None
            results['pruned'] = None
            return results

        route.append(closestCity)
        citiesToSearch.remove(closestCity)
        startCity = closestCity
        totalCost += closestDistance

    # If the route found isn't complete, run the alrogithm again with a different starting city
    if len(route) != len(cities) or route[-1].costTo(route[0]) == math.inf:
        print("Route not complete, running again with different starting city")
        return greedyTSP(cities, time_allowance=time_allowance, startIndex=startIndex + 1, startTime=startTime)

    solution = TSPSolution(route)
    endTime = time.time()

    results["cost"] = solution.cost
    results["time"] = endTime - startTime
    results["soln"] = solution
    results["count"] = 1
    results['max'] = None
    results['total'] = None
    results['pruned'] = None

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
        print(f"Switching to branch and bound after {self.minBSSFBeforeSwitch} solutions")

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
                    print(f"Found a better solution! (#{self._numSolutions}: {self._bssf})")

                    if self._numSolutions > self.minBSSFBeforeSwitch and self.searchByDepth:
                        self.searchByDepth = False
                        print("Switching to prioritizing lower bound over depth. "
                              f"Heapifying {len(self.priorityQueue)} nodes...)")
                        # Redo the heap with the new priority
                        self.priorityQueue = [(node.score, node) for score, node in self.priorityQueue]
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
                        heapq.heappush(self.priorityQueue, (newNode.depth * -1, newNode))

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
        results['max'] = maxQueueSize
        results['total'] = self._numNodes
        results['pruned'] = self._numPruned

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
