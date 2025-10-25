# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    initial_state = problem.get_start_state()
    frontier = util.Stack()
    initial_state = SearchNode(None, (initial_state, None, 0))
    frontier.push(initial_state)
    expanded_nodes = set()
    while not frontier.is_empty():
        current_node = frontier.pop()
        if current_node.state in expanded_nodes:
            continue
        expanded_nodes.add(current_node.state)
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        for successor in problem.get_successors(current_node.state): 
            successor = SearchNode(current_node, successor)
            if successor.state not in expanded_nodes:
                frontier.push(successor)

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    initial_state = problem.get_start_state()
    frontier = util.Queue()
    initial_state = SearchNode(None, (initial_state, None, 0))
    frontier.push(initial_state)
    expanded_nodes = set()
    while not frontier.is_empty():
        current_node = frontier.pop()
        if current_node.state in expanded_nodes:
            continue
        expanded_nodes.add(current_node.state)
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        for successor in problem.get_successors(current_node.state): 
            successor = SearchNode(current_node, successor)
            if successor.state not in expanded_nodes:
                frontier.push(successor)


def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"    
    from util import PriorityQueue

    start = problem.get_start_state()
    if problem.is_goal_state(start):
        return []

    frontier = PriorityQueue()          # items are just 'state'
    best_g = {start: 0}                 # best cost to each state
    path = {start: []}                  # action path to each state
    explored = set()

    # push start with priority 0
    frontier.push(start, 0)

    while not frontier.is_empty():
        state = frontier.pop()          # get the state with smallest g
        if state in explored:
            continue
        explored.add(state)

        # goal test on POP ensures optimality
        if problem.is_goal_state(state):
            return path[state]

        g = best_g[state]

        # expand
        for succ, action, step_cost in problem.get_successors(state):
            new_g = g + step_cost
            # if we found a cheaper path to succ, record and decrease-key
            if succ not in best_g or new_g < best_g[succ]:
                best_g[succ] = new_g
                path[succ] = path[state] + [action]
                frontier.update(succ, new_g)   # push if absent, lower priority if present

    return []  # no solution

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0




def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    start = problem.get_start_state()
    print("the start state is",start)
    if problem.is_goal_state(start):
        return []

    frontier = PriorityQueue()          # holds states with priority f = g + h
    best_g = {start: 0}                 # best known g-cost to each state
    path = {start: []}                  # actions to reach each state

    # push start with f = 0 + h(start)
    frontier.push(start, heuristic(start, problem))

    while not frontier.is_empty():
        state = frontier.pop()          # state with smallest f

        # If we popped a state we no longer have the best g for, skip it
        # (this handles outdated queue entries without needing an explored set)
        g = best_g[state]

        if problem.is_goal_state(state):
            return path[state]

        # expand
        for succ, action, step_cost in problem.get_successors(state):
            new_g = g + step_cost
            if succ not in best_g or new_g < best_g[succ]:
                best_g[succ] = new_g
                path[succ] = path[state] + [action]
                f = new_g + heuristic(succ, problem)
                frontier.update(succ, f)   # decrease-key or push if absent

    return []  # no solution

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search