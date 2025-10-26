# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        # The evaluation score starts with the score of the successor state.
        # This inherently includes rewards for eating food.
        evaluation_score = successor_game_state.get_score()
        
        # Get a list of the positions of the remaining food.
        food_list = new_food.as_list()
        
        # Penalize stopping, as it's generally not a productive move for Pacman.
        if action == Directions.STOP:
            evaluation_score -= 10
        
        # Add a significant bonus if a capsule is eaten in this move.
        # This encourages Pacman to eat capsules to gain an advantage over ghosts.
        if len(current_game_state.get_capsules()) > len(successor_game_state.get_capsules()):
            evaluation_score += 100
        
        # If there is still food on the board, calculate the distance to the nearest food pellet.
        # The score is increased by an amount inversely proportional to this distance.
        # This incentivizes Pacman to move towards the closest food.
        if food_list:
            min_food_distance = min([manhattan_distance(new_pos, food) for food in food_list])
            evaluation_score += 10.0 / (min_food_distance + 1)

        # Evaluate interactions with each ghost.
        for ghost_state in new_ghost_states:
            ghost_pos = ghost_state.get_position()
            distance_to_ghost = manhattan_distance(new_pos, ghost_pos)
            
            # Check if the ghost is scared.
            if ghost_state.scared_timer > 0:
                # If the ghost is scared, it's an opportunity.
                # We add a reward for being close to a scared ghost, encouraging Pacman to chase it.
                # The reward is higher the closer Pacman is.
                if distance_to_ghost <= ghost_state.scared_timer:
                    evaluation_score += 20.0 / (distance_to_ghost + 1)
            else:
                # If the ghost is not scared, it's a threat.
                # If Pacman is very close to a non-scared ghost, it's an extremely dangerous situation.
                # We return a very low score to avoid this move.
                if distance_to_ghost <= 1:
                    return -float('inf') # Extremely bad to be this close
                
                # Otherwise, apply a penalty that is inversely proportional to the distance.
                # This encourages Pacman to keep a safe distance from ghosts.
                evaluation_score -= 10.0 / (distance_to_ghost + 1)
        return evaluation_score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
         
        # This is the root of the minimax search, where we decide Pacman's best move.
        best_score=float("-inf")
        best_action=None
        actions = game_state.get_legal_actions(0)
        # Iterate through Pacman's possible first moves.
        for action in actions:
            pacman=0
            next_state = game_state.generate_successor(pacman,action)
            # For each move, calculate the score by starting the minimax recursion.
            # The first recursive call is for the first ghost (player 1) at depth 0.
            depth=0
            score=self.minimax(1,next_state,depth)
            # If the score for this action is better than our best score, update our choice.
            if score>best_score:
                best_score=score
                best_action=action
        return best_action
    
    
    def minimax(self, player,game_state,depth):
        """
        This is the recursive helper function that implements the minimax algorithm.
        - player: The index of the current agent (0 for Pacman, >0 for ghosts).
        - game_state: The current state of the game.
        - depth: The current depth in the search tree.
        """
        # Base case: if the game is over or the maximum depth is reached,
        # return the evaluation of the current state.
        if game_state.is_win() or game_state.is_lose() or depth >= self.depth:
            return self.evaluation_function(game_state)
        
        # Pacman's turn (Maximizer)
        if player==0:
            maxscore=float('-inf')
            # Iterate through all legal actions for Pacman.
            for action in game_state.get_legal_actions(player):
                next_state = game_state.generate_successor(player,action)
                # Recursively call minimax for the next player (the first ghost).
                score=self.minimax(1,next_state,depth)
                # Update the max score found so far.
                if score>maxscore:
                    maxscore=score
            return maxscore    
        else:
            # A ghost's turn (Minimizer)
            minscore=float('inf')
            # Iterate through all legal actions for the current ghost.
            for action in game_state.get_legal_actions(player):
                next_state = game_state.generate_successor(player,action)
                
                # If there are more ghosts to move 
                if player < game_state.get_num_agents()-1:
                    next_player=player+1
                    # recursively call minimax for the next ghost at the same depth.
                    score=self.minimax(next_player,next_state,depth)
                # If this is the last ghost's turn
                elif player==game_state.get_num_agents()-1:
                    next_depth=depth+1
                    next_player=0
                    # recursively call minimax for Pacman at the next depth level.
                    score=self.minimax(next_player,next_state,next_depth)
                
                # Update the min score found so far.
                if score<minscore:
                    minscore=score
            return minscore       
    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        # Get all legal actions for Pacman (agent 0).
        actions = game_state.get_legal_actions(0)
        if not actions:
             return None
        
        # Initialize alpha (the best score for the maximizer) to negative infinity
        # and beta (the best score for the minimizer) to positive infinity.
        alpha = float('-inf')
        beta = float('inf')
        best_action = None

        # This is the root of the search tree (a MAX node for Pacman).
        # We iterate through each action to find the one that maximizes the score.
        for action in actions:
            # For each possible action, calculate the value of the successor state.
            # The successor is evaluated by the first ghost (agent 1), which is a MIN node.
            successor = game_state.generate_successor(0, action)
            current_value = self.min_value(successor, 0, 1, alpha, beta)

            # Update alpha and the best action. Alpha represents the best score
            # that the maximizer (Pacman) can guarantee at this level or above.
            if current_value > alpha:
                alpha = current_value
                best_action = action
        # print(best_action)
        return best_action
    
    def max_value(self, state, depth, alpha, beta):
        """
        Calculates the value for a MAX node (Pacman's turn).
        - alpha: The best value found so far for the maximizer along the path to the root.
        - beta: The best value found so far for the minimizer along the path to the root.
        """
        # Base case: if the state is terminal or max depth is reached, return the evaluation score.
        if state.is_win() or state.is_lose() or depth >= self.depth:
            return self.evaluation_function(state)
        
        # Initialize the value for this MAX node to negative infinity.
        v = float('-inf')
        # For each legal action, find the value of the successor state.
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            # The value of this node is the maximum of the values of its MIN successors.
            v = max(v, self.min_value(successor, depth, 1, alpha, beta))
            
            # Pruning condition: If the current value `v` is greater than beta,
            # the parent MIN node will never choose this path, because it already
            # has an option that is better (lower) than `v`. We can stop exploring this branch.
            if v > beta:
                return v
            # Update alpha with the best value found so far for the maximizer at this node or below.
            alpha = max(alpha, v)
        return v
    
    def min_value(self, state, depth, agent, alpha, beta):
        """
        Calculates the value for a MIN node (a ghost's turn).
        - alpha: The best value found so far for the maximizer along the path to the root.
        - beta: The best value found so far for the minimizer along the path to the root.
        """
        # Base case: if the state is terminal or max depth is reached, return the evaluation score.
        if state.is_win() or state.is_lose() or depth >= self.depth:
            return self.evaluation_function(state)
        
        # Initialize the value for this MIN node to positive infinity.
        v = float('inf')
        # For each legal action, find the value of the successor state.
        for action in state.get_legal_actions(agent):
            successor = state.generate_successor(agent, action)
            
            # If this is the last ghost, the next player is Pacman (a MAX node at the next depth).
            if agent == state.get_num_agents() - 1:
                v = min(v, self.max_value(successor, depth + 1, alpha, beta))
            else:
                # Otherwise, the next player is the next ghost (another MIN node at the same depth).
                v = min(v, self.min_value(successor, depth, agent + 1, alpha, beta))
            
            # Pruning condition: If the current value `v` is less than alpha,
            # the parent MAX node will never choose this path, because it already
            # has an option that is better (higher) than `v`. We can stop exploring this branch.
            if v < alpha:
                return v
            # Update beta with the best value found so far for the minimizer at this node or below.
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
