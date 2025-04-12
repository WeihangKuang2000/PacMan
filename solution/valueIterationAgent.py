from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        for i in range(iters):
            for state in mdp.getStates():
                self.values[state] = 0.0

        for i in range(iters):
            new_value = {}
            for state in mdp.getStates():
                if self.getAction(state):
                    new_value[state] = self.getQValue(state, self.getAction(state))

            self.values = new_value

    def getPolicy(self, state):
        max_action = None

        if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state)
            max_value = -float("inf")

            for action in actions:
                value = self.getQValue(state, action)
                if max_value < value:
                    max_value = value
                    max_action = action

        return max_action

    def getQValue(self, state, action):
        # Q(s, a) = sum(T(s, a, s')) * [R(s, a, s') + y V(s')]
        # Q(s, a) =     probaility   *       value
        TransitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0

        for T_and_P in TransitionStatesAndProbs:
            reward = self.mdp.getReward(state, action, T_and_P[0])
            value = self.getValue(T_and_P[0])
            y = self.discountRate
            QValue += T_and_P[1] * (reward + y * value)

        return QValue

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        value = self.values.get(state)
        if value is None:
            return 0.0
        else:
            return value

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
