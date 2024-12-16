import util
from util import *
import heapq

class PriorityQueue:
    """Priority Queue implementation using a heap."""
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        return heapq.heappop(self.heap)[1]

    def isEmpty(self):
        return len(self.heap) == 0

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    from util import Stack  # Make sure to import Stack

    # The path we follow to reach a goal
    currPath = []           
    # The current state (position)
    currState = problem.getStartState()    

    # If the starting state is already a goal state, return an empty path
    if problem.isGoalState(currState):
        return currPath

    # Use a stack for DFS (LIFO order)
    frontier = Stack()
    frontier.push((currState, currPath))  # Push the start state and its path
    
    # Set to keep track of explored states
    explored = set()

    while not frontier.isEmpty():
        currState, currPath = frontier.pop()  # Pop the state and path from the stack

        # If the state is the goal, return the path
        if problem.isGoalState(currState):
            return currPath
        
        # Mark the state as explored
        explored.add(currState)
        
        # Explore all successors of the current state
        for s in problem.getSuccessors(currState):
            if s[0] not in explored:  # If not yet explored
                frontier.push((s[0], currPath + [s[1]]))  # Push the successor and its path

    # If no solution is found, return an empty list
    return []  


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue  # Make sure to import Queue

    # Initialize the frontier with the start state and an empty path
    frontier = Queue()
    start_state = problem.getStartState()
    frontier.push((start_state, []))  # Push state with path
    
    # Set to keep track of explored states
    explored = set()

    while not frontier.isEmpty():
        curr_state, curr_path = frontier.pop()  # Pop the state and path from the queue

        # If the state is the goal, return the path
        if problem.isGoalState(curr_state):
            return curr_path

        if curr_state not in explored:
            explored.add(curr_state)
            # Explore all successors of the current state
            for successor, action, _ in problem.getSuccessors(curr_state):
                if successor not in explored:
                    frontier.push((successor, curr_path + [action]))  # Push successor with updated path

    # If no solution is found, return an empty list
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue  # Make sure to import the correct class

    # Initialize the frontier with the start state and an empty path
    frontier = PriorityQueue()
    start_state = problem.getStartState()
    frontier.push((start_state, []), 0)  # Push state with path and cost (priority)
    
    # Set to keep track of explored states
    explored = set()

    # To track the cost of reaching each state
    cost_so_far = {start_state: 0}

    while not frontier.isEmpty():
        curr_state, curr_path = frontier.pop()  # Pop the state with the lowest cost

        # If the state is the goal, return the path
        if problem.isGoalState(curr_state):
            return curr_path

        # If the state has not been explored
        if curr_state not in explored:
            explored.add(curr_state)
            # Explore all successors of the current state
            for successor, action, step_cost in problem.getSuccessors(curr_state):
                new_cost = cost_so_far[curr_state] + step_cost
                if successor not in cost_so_far or new_cost < cost_so_far[successor]:
                    cost_so_far[successor] = new_cost
                    frontier.push((successor, curr_path + [action]), new_cost)  # Push successor with updated cost

    return []  # Return empty if no solution is found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue  # Make sure to import the correct class

    # Initialize the frontier with the start state and an empty path
    frontier = PriorityQueue()
    start_state = problem.getStartState()
    frontier.push((start_state, []), 0)  # Push state with path and cost (priority)
    
    # Set to keep track of explored states
    explored = set()

    # To track the cost of reaching each state
    cost_so_far = {start_state: 0}

    while not frontier.isEmpty():
        curr_state, curr_path = frontier.pop()  # Pop the state with the lowest combined cost

        # If the state is the goal, return the path
        if problem.isGoalState(curr_state):
            return curr_path

        # If the state has not been explored
        if curr_state not in explored:
            explored.add(curr_state)
            # Explore all successors of the current state
            for successor, action, step_cost in problem.getSuccessors(curr_state):
                new_cost = cost_so_far[curr_state] + step_cost
                total_cost = new_cost + heuristic(successor, problem)
                if successor not in cost_so_far or new_cost < cost_so_far[successor]:
                    cost_so_far[successor] = new_cost
                    frontier.push((successor, curr_path + [action]), total_cost)  # Push successor with updated cost

    return []  # Return empty if no solution is found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
