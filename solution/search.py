"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    from pacai.util.stack import Stack

    start_state = problem.startingState()
    node_stack = Stack()
    path_stack = Stack()
    visited = []
    paths = []
    node_stack.push(start_state)
    path_stack.push(paths)

    while node_stack:
        state = node_stack.pop()
        paths = path_stack.pop()
        if problem.isGoal(state):
            break
        elif state in visited:
            continue
        else:
            visited.append(state)
            for successor in problem.successorStates(state):
                node_stack.push(successor[0])
                path_stack.push(paths + [successor[1]])

    # print(len(paths))
    return paths

    # raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    from pacai.util.queue import Queue

    start_state = problem.startingState()
    node_queue = Queue()
    path_queue = Queue()
    visited = []
    paths = []
    node_queue.push(start_state)
    path_queue.push(paths)

    while node_queue:
        state = node_queue.pop()
        paths = path_queue.pop()
        if problem.isGoal(state):
            break
        elif state in visited:
            continue
        else:
            visited.append(state)
            for successor in problem.successorStates(state):
                node_queue.push(successor[0])
                path_queue.push(paths + [successor[1]])

    return paths

    # raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    from pacai.util.priorityQueue import PriorityQueue

    start_state = problem.startingState()
    node_queue = PriorityQueue()
    path_queue = PriorityQueue()
    visited = []
    paths = []
    node_queue.push((start_state, 0), -1)
    path_queue.push((start_state, []), -1)

    while node_queue:
        state, costs = node_queue.pop()
        state, paths = path_queue.pop()
        if problem.isGoal(state):
            break
        elif state in visited:
            continue
        else:
            visited.append(state)
            for successor in problem.successorStates(state):
                newcost = costs + successor[2]
                node_queue.push((successor[0], newcost), newcost)
                path_queue.push((successor[0], paths + [successor[1]]), newcost)

    return paths
    # raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    from pacai.util.priorityQueue import PriorityQueue

    start_state = problem.startingState()
    node_queue = PriorityQueue()
    path_queue = PriorityQueue()
    visited = []
    paths = []
    node_queue.push((start_state, 0), -1)
    path_queue.push((start_state, []), -1)

    while node_queue:
        state, costs = node_queue.pop()
        state, paths = path_queue.pop()
        if problem.isGoal(state):
            break
        elif state in visited:
            continue
        else:
            visited.append(state)
            for successor in problem.successorStates(state):
                newcost = costs + successor[2]
                hn = newcost + heuristic(successor[0], problem)
                node_queue.push((successor[0], newcost), hn)
                path_queue.push((successor[0], paths + [successor[1]]), hn)

    return paths
    # raise NotImplementedError()
