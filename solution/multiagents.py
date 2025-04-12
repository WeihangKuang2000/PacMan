import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        # *** Your Code Here ***
        from pacai.core.distance import manhattan
        from pacai.core.distance import euclidean

        score = successorGameState.getScore()
        minFood = float("inf")
        maxFood = -float("inf")

        for food in successorGameState.getFood().asList():
            m = manhattan(newPosition, food)
            e = euclidean(newPosition, food)
            mind = min(m, e)
            maxd = max(m, e)
            if minFood > mind:
                minFood = mind
            if maxFood < maxd:
                maxFood = maxd

        for T, ghost in zip(newScaredTimes, successorGameState.getGhostPositions()):
            m = manhattan(newPosition, ghost)
            e = euclidean(newPosition, ghost)
            if T > 0:
                if minFood > m:
                    minFood = m
            elif (m <= 2 or e <= 2):
                return -10000

        minFood = 1.0 / minFood
        maxFood = 1.0 / maxFood

        if newPosition in currentGameState.getFood().asList():
            return score + minFood + maxFood + 1000
        else:
            return score + minFood + maxFood

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):

        def minimax(gameState, index, depth):
            if index == 0:
                depth -= 1
                return max_value(gameState, index, depth)
            else:
                return min_value(gameState, index, depth)

        def max_value(gameState, index, depth):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return None, self.getEvaluationFunction()(gameState)

            maxvalue = -float("inf")
            best_action = "none"
            # keep track of index as recursive function mess up with index
            nextIndex = (index + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(index):
                if action != "Stop":
                    nextState = gameState.generateSuccessor(index, action)
                    _, value = min_value(nextState, nextIndex, depth)
                    if value > maxvalue:
                        best_action = action
                        maxvalue = value
            return best_action, maxvalue

        def min_value(gameState, index, depth):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return None, self.getEvaluationFunction()(gameState)

            minvalue = float("inf")
            nextIndex = (index + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(index):
                if action != "Stop":
                    nextState = gameState.generateSuccessor(index, action)
                    _, value = minimax(nextState, nextIndex, depth)
                    if value < minvalue:
                        minvalue = value
            return _, minvalue

        # action retuen (action, value) we take action so [0]
        action = minimax(gameState, self.index, self.getTreeDepth())[0]
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):  # copy from MiniMax and modify

        def alpha_beta(gameState, index, depth, alpha, beta):
            if index == 0:
                depth -= 1
                return max_value(gameState, index, depth, alpha, beta)
            else:
                return min_value(gameState, index, depth, alpha, beta)

        def max_value(gameState, index, depth, alpha, beta):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return None, self.getEvaluationFunction()(gameState)

            best_action = "none"
            nextIndex = (index + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(index):
                if action != "Stop":
                    nextState = gameState.generateSuccessor(index, action)
                    _, value = min_value(nextState, nextIndex, depth, alpha, beta)
                    if value > alpha:
                        best_action = action
                        alpha = value

                if alpha >= beta:
                    return best_action, alpha

            return best_action, alpha

        def min_value(gameState, index, depth, alpha, beta):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return None, self.getEvaluationFunction()(gameState)

            nextIndex = (index + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(index):
                if action != "Stop":
                    nextState = gameState.generateSuccessor(index, action)
                    _, value = alpha_beta(nextState, nextIndex, depth, alpha, beta)
                    if value < beta:
                        beta = value

                if beta <= alpha:
                    return _, beta

            return _, beta

        # action retuen (action, value) we take action so [0]
        depth = self.getTreeDepth()
        action = alpha_beta(gameState, self.index, depth, -float("inf"), float("inf"))[0]
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):  # copy from MiniMax and modify

        def expecti(gameState, index, depth):
            if index == 0:
                depth -= 1
                return max_value(gameState, index, depth)
            else:
                return exp_value(gameState, index, depth)

        def max_value(gameState, index, depth):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return None, self.getEvaluationFunction()(gameState)

            maxvalue = -float("inf")
            best_action = "none"
            nextIndex = (index + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(index):
                if action != "Stop":
                    nextState = gameState.generateSuccessor(index, action)
                    _, value = exp_value(nextState, nextIndex, depth)
                    if value is not None and value > maxvalue:
                        best_action = action
                        maxvalue = value

            return best_action, maxvalue

        def exp_value(gameState, index, depth):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return None, self.getEvaluationFunction()(gameState)

            nextIndex = (index + 1) % gameState.getNumAgents()
            action_num = 0
            if "Stop" in gameState.getLegalActions(index):
                action_num = len(gameState.getLegalActions(index)) - 1
            else:
                action_num = len(gameState.getLegalActions(index))
            propability = float(1.0 / action_num)
            weights = 0.0
            for action in gameState.getLegalActions(index):
                if action != "Stop":
                    nextState = gameState.generateSuccessor(index, action)
                    _, value = expecti(nextState, nextIndex, depth)
                    if value is not None:
                        weights += value * propability

            return _, weights

        # action retuen (action, value) we take action so [0]
        action = expecti(gameState, self.index, self.getTreeDepth())[0]
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: The evaluation function is linear combination of features. I think closest food
    is important to consider, and ghost distance and scare time. Also remaining_food and the score
    we have right now. As this evaluation function decide what action to do next, we will add on
    to the current score. We choose the closest food or capsule. We want to eat the capsule as well
    to increase our score by eating a ghost. But we only want to ear a capsule when we close to it
    and worth to go to eat it. We also take into account the terminal states to make sure we aim to
    win the game. Scared time and ghost distance tell us should we chase onto the ghost or not.
    And we reduce point by food remaining, as we want to eat all the food as quickly as possible.
    Overall, I think closest food > ghost distance > remaining food > the rest
    """

    from pacai.core.distance import manhattan
    from pacai.core.distance import euclidean

    position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    # Feature 1: Distance to the nearest food pellet
    minFood = float("inf")
    for food in foodList:
        m = manhattan(position, food)
        e = euclidean(position, food)
        m = min(m, e)
        if minFood > m:
            minFood = m

    # Feature 2: Distance to the closest ghost and whether the ghost is scared or not
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
    ghost_dis = float("inf")
    scared_time = 0
    for T, ghost in zip(newScaredTimes, currentGameState.getGhostPositions()):
        m = manhattan(position, ghost)
        e = euclidean(position, ghost)
        if T > 0:
            scared_time += T
            if ghost_dis > m:
                ghost_dis = m
        elif (m < 2 or e < 2):
            ghost_dis = -100000
            return ghost_dis
    ghost_dis = 1.0 / ghost_dis

    # Feature 3: Current game score
    now_score = currentGameState.getScore()

    # Feature 4: Number of remaining food pellets
    remaining_food = currentGameState.getNumFood()

    # Feature 5: Distance to the nearest power capsule
    for capsule in capsules:
        m = manhattan(position, capsule)
        if minFood > m:
            minFood = m
    minFood = 1.0 / minFood

    # Feature 6 : Terminal state bonuses
    terminal = 0
    if currentGameState.isLose():
        terminal = -10000
    elif currentGameState.isWin():
        terminal = 10000

    # Adjusting weights and combining fectures
    if position in currentGameState.getFood().asList():
        return now_score + 5 * minFood + 4 * ghost_dis + terminal + 3.5 * remaining_food + \
            scared_time + 1000
    else:
        return now_score + 5 * minFood + 4 * ghost_dis + terminal + 3.5 * remaining_food + \
            scared_time

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
