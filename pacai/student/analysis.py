"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I try to work on with Discount first, but discount won't
    make the agent successfully cross the bridge, so I try to
    work on the Noise only. When Noise is 0, the agent cross
    the bridge. I think this is becasue we make sure the agent
    ends up in an intended successor state when they perform
    an action. 0 make it 100%
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    I try to decrease the disount so the agent become more short-
    signed, focus more on immediate score. Also, I try to decrease
    the noise, to make the agent ends up in an intended successor
    state. Also, I decrease the living reward, so the agent won't
    want to live too long
    """

    answerDiscount = 0.01
    answerNoise = 0.0
    answerLivingReward = -0.3

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    I first decrease the living reward, so the agent won't
    want to live too long. And then I decrease the discount,
    so the agent to focus on immediate score. I choose not to
    change the noise. Since the original noise did pretty
    good on avoiding the cliff
    """

    answerDiscount = 0.299
    answerNoise = 0.3
    answerLivingReward = -0.8

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    I think for this one just change the noise to 0 will be enough.
    So every move that the agent do will end up in an intended
    successor state.
    """

    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    I do not think I need to modifie anything for 3d. The default
    already did pretty good.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    First I make noise very big so every step the agent
    make is uncertain. Also, make the discount very small,
    so the agent can not see the reward.
    """

    answerDiscount = 0.001
    answerNoise = 1
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    I first try on different epsilon to make the agent
    to explore more state. Also, I try to increase the
    learning rate to see if it can learn all state faster,
    but I can not make it learn from all state. It only focus
    on few states.
    """

    # answerEpsilon = 0.3
    # answerLearningRate = 0.5

    # return answerEpsilon, answerLearningRate
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
