from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'EatAgent',
        second = 'DefensiveAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        eval(first)(firstIndex),
        eval(second)(secondIndex),
    ]

class EatAgent(CaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):

        # Collect legal moves.
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions.
        scores = []
        for action in legalMoves:
            features = self.getFeatures(gameState, action)
            weights = self.getWeights(gameState, action)
            score = sum(features[feature] * weights[feature] for feature in features)
            if score is None:
                score = 0
            scores.append(score)

        # Choose one of the best actions.
        # scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def getFeatures(self, gameState, action):
        features = {}
        successor = gameState.generateSuccessor(self.index, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = 1.0 / minDistance

        myState = successor.getAgentState(self.index)

        if myState.isPacman():
            opp_index = self.getOpponents(gameState)
            ghostStates = [successor.getAgentState(i) for i in opp_index]
            for g in ghostStates:
                if g.isPacman():
                    ghostStates.remove(g)
            ghostDistances = [self.getMazeDistance(myPos, ghostState.getPosition())
                    for ghostState in ghostStates]
            scaredTimes = [ghostState.getScaredTimer() for ghostState in ghostStates]

            for i, ghostDistance in enumerate(ghostDistances):
                if ghostDistance <= 1:
                    if scaredTimes[i] < 1:
                        features['distanceToGhost'] = 1.0 / ghostDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': 10,
            'distanceToGhost': -10000
        }

class DefensiveAgent(CaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):

        # Collect legal moves.
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions.
        scores = []
        for action in legalMoves:
            features = self.getFeatures(gameState, action)
            weights = self.getWeights(gameState, action)
            score = sum(features[feature] * weights[feature] for feature in features)
            if score is None:
                score = 0
            scores.append(score)

        # scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def getFeatures(self, gameState, action):
        features = {}

        successor = gameState.generateSuccessor(self.index, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0.1
        # print(myPos)
        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        else:
            # dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            closest_enemies = None
            dists = float('inf')
            for e in enemies:
                d = self.getMazeDistance(myPos, e.getPosition())
                if dists > d:
                    dists = d
                    closest_enemies = e
            features['invaderDistance'] = dists

            foodList = self.getFood(successor).asList()
            foodD = [self.getMazeDistance(closest_enemies.getPosition(), f) for f in foodList]
            # print(min(foodD))
            features['invaderclosestfood'] = min(foodD)
            # features['invaderDistance'] = min(dists)

        if (action == Directions.STOP and len(invaders) != 0):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        scaredTimes = myState.getScaredTimer()
        if scaredTimes != 0 and features['invaderDistance'] <= 1:
            features['invaderDistance'] += 100

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -20,
            'stop': -100,
            'reverse': -2,
            'invaderclosestfood': -18
        }
