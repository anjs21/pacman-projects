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

    # Early exit if the start state is the goal.
    if problem.is_goal_state(initial_state):
        return []

    # The frontier is a LIFO Stack to explore the deepest nodes first.
    frontier = util.Stack()

    # The initial node has no parent, no action, and zero cost.
    initial_state = SearchNode(None, (initial_state, None, 0))
    frontier.push(initial_state)

    # 'expanded_nodes' is a set to keep track of states we have already visited
    # to avoid cycles and redundant computations in this graph search.
    expanded_nodes = set()

    while not frontier.is_empty():
        current_node = frontier.pop()

        # If we have already expanded this state, skip it.
        if current_node.state in expanded_nodes:
            continue

        # Mark the current state as expanded.
        expanded_nodes.add(current_node.state)

        # If the current node is the goal, we have found a solution.
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        
        # Get successors and add them to the frontier if they haven't been expanded.
        for successor in problem.get_successors(current_node.state): 
            successor = SearchNode(current_node, successor)
            if successor.state not in expanded_nodes:
                frontier.push(successor)
    return []  # no solution found

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    initial_state = problem.get_start_state()

    # Early exit if the start state is the goal.
    if problem.is_goal_state(initial_state):
        return []

    # The frontier is a FIFO Queue to explore level by level.
    frontier = util.Queue()

    # The initial node has no parent, no action, and zero cost.
    initial_state = SearchNode(None, (initial_state, None, 0))
    frontier.push(initial_state)

    # 'expanded_nodes' stores states that have been visited to prevent cycles.
    expanded_nodes = set()

    while not frontier.is_empty():
        current_node = frontier.pop()

        # If we have already expanded this state, skip it.
        if current_node.state in expanded_nodes:
            continue
        # Mark current state as expanded.
        expanded_nodes.add(current_node.state)

        # If the current node is the goal, we have found a solution.
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        
        # Get successors and add them to the frontier if they haven't been expanded.
        for successor in problem.get_successors(current_node.state): 
            successor = SearchNode(current_node, successor)
            if successor.state not in expanded_nodes:
                frontier.push(successor)
    return []  # no solution found


def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"    

    start = problem.get_start_state()

    # Early exit if the start state is the goal.
    if problem.is_goal_state(start):
        return []

    # The frontier is a Priority Queue, ordered by the path cost (g-value).
    # Nodes with lower path cost are explored first.
    frontier = util.PriorityQueue()
    explored = set()

    # Push the start node with a priority equal to its cost (0).
    initial_state = SearchNode(None, (start, None, 0))
    frontier.push(initial_state, initial_state.cost)

    while not frontier.is_empty():

        # Pop the node with the lowest path cost from the frontier.
        current_node = frontier.pop()
        if current_node.state in explored:
            continue
        explored.add(current_node.state)

        # Goal test is performed when a node is selected for expansion.
        # This ensures optimality for UCS because we always expand the lowest-cost path first.
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()

        # Expand the current node and add its successors to the frontier.
        for successor in problem.get_successors(current_node.state):
            successor = SearchNode(current_node, successor)
            if successor.state not in explored:
                # Use update to add the node to the frontier or update its priority
                # if a shorter path to it is found.
                # update(priority, item) updates the priority of item if the new
                # priority is lower than the current one. If the item is not in the queue,
                # it is added with the given priority. 
                # (so, we don't need to check if it's already in the frontier)
                frontier.update(successor, successor.cost)

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

    start = problem.get_start_state()
    # Early exit if the start state is the goal.
    if problem.is_goal_state(start):
        return []

    # The frontier is a Priority Queue ordered by f(n) = g(n) + h(n),
    # where g(n) is the path cost and h(n) is the heuristic value.
    frontier = util.PriorityQueue()
    explored = set()

    # Push the start node with priority f(start) = g(start) + h(start).
    # g(start) is 0.
    initial_state = SearchNode(None, (start, None, 0))
    frontier.push(initial_state, initial_state.cost + heuristic(initial_state.state, problem))

    while not frontier.is_empty():
        # Pop the node with the lowest f-value from the frontier.
        current_node = frontier.pop()
        if current_node.state in explored:
            continue
        explored.add(current_node.state)

        # Goal test is performed when a node is selected for expansion.
        # If the heuristic is consistent, this guarantees an optimal solution.
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()

        # Expand the current node and add its successors to the frontier.
        for successor in problem.get_successors(current_node.state):
            successor = SearchNode(current_node, successor)
            if successor.state not in explored:
                # The priority for the successor is its f-value: g(successor) + h(successor).
                # g(successor) is successor.cost.
                priority = successor.cost + heuristic(successor.state, problem)

                # Use update to add the node or update its priority if a better path is found.
                # If not in the frontier, it is added with the given priority.
                frontier.update(successor, priority)

    return []  # no solution

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search