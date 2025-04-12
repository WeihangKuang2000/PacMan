from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.Q_values = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        if (state, action) in self.Q_values:
            return self.Q_values[(state, action)]
        else:
            return 0.0

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        if len(self.getLegalActions(state)) == 0:
            return 0.0
        value = -float("inf")
        for action in self.getLegalActions(state):
            V = self.getQValue(state, action)
            if value < V:
                value = V
        return value

    def getAction(self, state):
        from pacai.util.probability import flipCoin
        import random
        actions = self.getLegalActions(state)
        # use self.epsilon because this is epsilon-greedy action selection
        ran = flipCoin(self.epsilon)
        if ran:
            return random.choice(actions)
        else:
            return self.getPolicy(state)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        import random

        if len(self.getLegalActions(state)) == 0:
            return None

        best_A = []
        best_V = self.getValue(state)

        for action in self.getLegalActions(state):
            V = self.getQValue(state, action)
            if V == best_V:
                best_A.append(action)

        if len(best_A) > 1:
            return random.choice(best_A)
        else:
            return best_A[0]

    def update(self, state, action, nextState, reward):
        # Q-Learning:
        # Q(s,a) = (1 - alpha)Q(s, a) + alpha * [sample]
        # sample = R(s, a, s') + y max(Q(s', a'))
        alpha = self.getAlpha()
        discountRate = self.getDiscountRate()
        qv = self.getQValue(state, action)
        next_v = self.getValue(nextState)

        # self.getValue return the max value
        new_q = (1 - alpha) * qv + alpha * (reward + discountRate * next_v)

        self.Q_values[(state, action)] = new_q

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = {}

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        # if self.episodesSoFar == self.numTraining:
        #     # You might want to print your weights here for debugging.
        #     # *** Your Code Here ***
        #     print(self.weights)
        #     # raise NotImplementedError()

    def getQValue(self, state, action):
        # Q(state, action) = w * (dotProduct) featureVector
        features = self.featExtractor.getFeatures(self, state, action)

        Q = 0
        for f in features:
            if f in self.weights:
                Q += self.weights[f] * features[f]
            else:
                Q += features[f]
        return Q

    def update(self, state, action, nextState, reward):
        # correction = (R(s, a) + y * V(s')) - Q(s, a)
        # wi = wi + alpha * correction * fi(s, a)
        features = self.featExtractor.getFeatures(self, state, action)
        y = self.discountRate
        V = self.getValue(nextState)
        QV = self.getQValue(state, action)
        correction = reward + y * V - QV
        for f in features:
            if f in self.weights:
                self.weights[f] += self.alpha * correction * features[f]
            else:
                self.weights[f] = self.alpha * correction * features[f]
