"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        # *** Your Code Here ***
        self._numExpanded = 0

    def startingState(self):
        corners = set(self.corners)  # save target corners into our state
        # so state[0] is the position, and state[1] include corners not went
        return (self.startingPosition, corners)

    def isGoal(self, state):
        node = state[0]  # where we are now
        corners = state[1]  # corners we did not go
        if node in corners:  # if node is corner we never go
            corners.remove(node)  # remove it so we can check how many corners left
        return len(corners) == 0

    def successorStates(self, state):
        """
        Returns successor states, the actions they require, and a constant cost of 1.
        """
        # code below learn from PositionSearchProbelm and FoodSearchProblem
        successors = []
        from pacai.core.directions import Directions
        for direction in Directions.CARDINAL:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:  # if not hit wall
                cornerstogo = state[1].copy()
                if (nextx, nexty) in cornerstogo:  # if nextState is a corner not went yet
                    cornerstogo.remove((nextx, nexty))  # remove itself from cornerstogo
                successor = (((nextx, nexty), cornerstogo), direction, 1)
                successors.append(successor)

        self._numExpanded += 1
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    cost_all_corners = heuristic.null(state, problem)  # Default to trivial solution

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    not_visited_corner = state[1].copy()  # copy of state[1](corners not visited)

    cost_all_corners = 0
    node = state[0]
    while not_visited_corner:  # calculate total cost access all corner, start from the least cost
        least_corner = (0, 0)
        cost = float('inf')
        for corner in not_visited_corner:
            cost_for_one = manhattan(node, corner)
            if cost > cost_for_one:
                cost = cost_for_one
                least_corner = corner
        not_visited_corner.remove(least_corner)  # remove the least cost corner
        node = least_corner  # make the least cost corner where we are now, and go to next one
        cost_all_corners += cost  # for total_cost to all left corner

    return cost_all_corners

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state
    # *** Your Code Here ***
    foodlist = foodGrid.asList()
    walllist = problem.walls.asList()

    # try to get the real distance concider wall, search the path with wall in place
    def maze_dis(position, food, walllist):
        from pacai.util.queue import Queue
        queue = Queue()
        queue.push((position, 0))  # 0 for cost to the food
        visited = set()
        while queue:
            node, cost = queue.pop()
            if node == food:
                return cost
            if node not in visited:
                visited.add(node)
                x, y = node
                # successors are the four node next to current position
                successors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for s in successors:
                    if s not in walllist:  # if it is wall, then we can not go
                        queue.push((s, cost + 1))
        return 0

    dist = 0
    for food in foodlist:
        if (position, food) not in problem.heuristicInfo:  # we save it increase process time
            problem.heuristicInfo[(position, food)] = maze_dis(position, food, walllist)
        food_dist = problem.heuristicInfo[(position, food)]
        # becuase we need to consider the farthest food
        if dist < food_dist:  # take the largest cost so never under estimate
            dist = food_dist
    return dist

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        # raise NotImplementedError()
        from pacai.student import search
        return search.uniformCostSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):

        def manhattan(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        foodcost = set()
        for food in self.food.asList():
            foodcost.add((manhattan(state, food), food))
        # we got all the cost for each food, since we want to use it
        # for findPathToClosestDot, we take the min cost(closest) dot over all
        _, goal = min(foodcost)
        if state == goal:
            return True
        else:
            return False

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
