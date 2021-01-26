# multiAgents.py
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

from util import manhattanDistance
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

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        paraFood = 1
        paraFood2 = 1
        paraGhost = 1
        """
            I just played with numbers to optimize the result. First I used large numbers like 100 or -500 but after
            many attempt I only got 3/4 since average score was around 500-600. Then I decided to use small numbers 
            like 1/ distances and I got pretty good results. 
        """
        # I check length of food list because I use min() and max() function and got error when they are empty.

        if len(newFood) != 0:
            distFood = manhattanDistance(newPos, min(newFood))
            paraFood = 1 / distFood
            distFood2 = manhattanDistance(newPos, max(newFood))
            paraFood2 = 1 / distFood2
        distGhost = manhattanDistance(newPos, min(successorGameState.getGhostPositions()))

        if distGhost < 2 and newScaredTimes[0] < 3:  # distGhost is very sensitive, the best results are at 2
            paraGhost = -distGhost
        elif distGhost < 2 and newScaredTimes[0] > 2:
            paraGhost = 1 / distGhost


        return successorGameState.getScore() + paraGhost + paraFood + paraFood2


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def maxValue(self, gameState, agentIndex, depth):
        """
        :param gameState: current game state
        :param agentIndex: current index of agent. Pacman (maximizer) is always 0
        :param depth: current depth
        :return: A list that contains an action(max) and a value that correspond to that action
        :does: Iterate through successor state to find maximized action. Since this is a recursive function, I thought
               I do not need to it will be better if I pass around value and action correspond to this value in a list
               so that look for the action that matches with this value.
        """
        # Termination, I used depth == 0 because I decrease the depth after last agent in the current depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), None]

        legalActions = gameState.getLegalActions(agentIndex)
        scoreAction = []
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action)
            value = self.minValue(succ, agentIndex + 1, depth)[0]
            scoreAction.append([value, action])
        return max(scoreAction)

    def minValue(self, gameState, agentIndex, depth):
        """
        :param gameState: current game state
        :param agentIndex: current index of agent. Pacman (maximizer) is always 0
        :param depth: current depth
        :return: A list that contains an action(min) and a value that correspond to that action
        :does: Iterate through successor state to find minimized action
        """
        # Termination, I used depth == 0 because I decrease the depth after last agent in the current depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), None]

        legalActions = gameState.getLegalActions(agentIndex)
        numOfAgents = gameState.getNumAgents()
        # We need to use module to get next agent since we do not know how many agents are in the game
        newAgentIndex = (agentIndex + 1) % numOfAgents
        scoreAction = []
        # If the agent is the last agent(ghost) at the current depth then decrease depth
        # after last ghost, new agent is pacman so we need to call maxValue function
        if agentIndex == numOfAgents - 1:
            nextDepth = depth - 1  # decrease the depth and new agent will be pacman
            for action in legalActions:
                succ = gameState.generateSuccessor(agentIndex, action)
                value = self.maxValue(succ, newAgentIndex, nextDepth)[0]
                scoreAction.append([value, action])
        else:
            nextDepth = depth
            for action in legalActions:
                succ = gameState.generateSuccessor(agentIndex, action)
                value = self.minValue(succ, newAgentIndex, nextDepth)[0]
                scoreAction.append([value, action])
        return min(scoreAction)

    def miniMax(self, gameState, agentIndex, depth):

        """
        :param gameState: game state
        :param agentIndex: agents index
        :param depth: current depth
        :return: an action that minimize the score if passed agent is a minimizer or an action that miximize the score
         if passed agent is a maximizer.
        """

        # If agent is maximizer (pacman's index is always 0) then call maxValue function
        # Else call minValue function
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[1]
        else:
            return self.minValue(gameState, agentIndex, depth)[1]

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Call the miniMax function with gameState, depth and currentAgentIndex =0 (there is pacman at the root)
        return self.miniMax(gameState, 0, self.depth)

        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agentIndex, depth, a, b):
        """
        :param gameState: current game state
        :param agentIndex: current index of agent. Pacman (maximizer) is always 0
        :param depth: current depth
        :param a: current alpha - as a list first element is score and second element is action
        :param b: current beta
        :return: A list that contains an action(max) and a value that correspond to that action
        :does: Iterate through successor state to find maximized action, also passes around alpha and beta values to
               prune the tree.
        """

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), None]

        legalActions = gameState.getLegalActions(agentIndex)
        value = [float("-inf"), None]
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action)
            value = max(value, [self.minValue(succ, agentIndex + 1, depth, a, b)[0], action])
            if value[0] > b[0]:
                return value
            a = max(a, value)

        return value

    def minValue(self, gameState, agentIndex, depth, a, b):
        """
        :param gameState: current game state
        :param agentIndex: current index of agent. Pacman (maximizer) is always 0
        :param depth: current depth
        :param a: alpha value as a list
        :param a: beta value as a list
        :return: A list that contains an action(min) and a value that correspond to that action
        :does: Iterate through successor state to find minimized action, it also passes a and b values and compare the
        results if it is smaller then alpha then it does not expands any more nodes,
        """

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), None]

        numOfAgents = gameState.getNumAgents()
        # We need to use module to get next agent since we do not know agent agents are there in the game
        newAgentIndex = (agentIndex + 1) % numOfAgents
        value = [float("inf"), None]

        for action in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numOfAgents - 1:
                nextDepth = depth - 1  # decrease the depth and next agent will be pacman
                value = min(value, [self.maxValue(succ, newAgentIndex, nextDepth, a, b)[0], action])
            else:
                nextDepth = depth
                value = min(value, [self.minValue(succ, newAgentIndex, nextDepth, a, b)[0], action])

            if value[0] < a[0]:
                return value
            else:
                b = min(b, value)
        return value

    def miniMaxAlphaBeta(self, gameState, agentIndex, depth):
        # If agent is maximizer (pacman's index is always 0) then call maxValue function
        # Else call minValue function
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, [float("-inf"), None], [float("inf"), None])[1]
        else:
            return self.minValue(gameState, agentIndex, depth, [float("-inf"), None], [float("inf"), None])[1]

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        return self.miniMaxAlphaBeta(gameState, 0, self.depth)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, gameState, agentIndex, depth):
        """
        :param gameState: current game state
        :param agentIndex: current index of agent. Pacman (maximizer) is always 0
        :param depth: current depth
        :return: A list that contains an action and a value that correspond to that action
        :does: Iterate through successor state to find maximized action
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), None]

        legalActions = gameState.getLegalActions(agentIndex)
        scoreAction = []
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action)
            value = self.minValue(succ, agentIndex + 1, depth)[0]
            scoreAction.append([value, action])
        return max(scoreAction)

    def minValue(self, gameState, agentIndex, depth):
        """
        :param gameState: current game state
        :param agentIndex: current index of agent. Pacman (maximizer) is always 0
        :param depth: current depth
        :return: A list that contains an action(min) and a value that correspond to that action
        :does: Iterate through successor state to find optimized action
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [self.evaluationFunction(gameState), None]

        numOfAgents = gameState.getNumAgents()
        # We need to use module to get next agent since we do not know agent agents are there in the game
        newAgentIndex = (agentIndex + 1) % numOfAgents
        scoreAction = []
        # If the agent is the last agent(ghost) at the current depth then decrease depth
        # after last ghost, new agent is pacman so we need to call maxValue function

        for action in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numOfAgents - 1:
                nextDepth = depth - 1  # decrease the depth and next agent will be pacman
                value = self.maxValue(succ, newAgentIndex, nextDepth)[0]
                scoreAction.append([value, action])
            else:
                nextDepth = depth
                value = self.minValue(succ, newAgentIndex, nextDepth)[0]
                scoreAction.append([value, action])
        res = 0
        for each in scoreAction:
            res += each[0]
        return [res / len(scoreAction)]

    def expectiMax(self, gameState, agentIndex, depth):

        # If agent is maximizer (pacman's index is always 0) then call maxValue function
        # Else call minValue function
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[1]
        else:
            return self.minValue(gameState, agentIndex, depth)[1]

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.expectiMax(gameState, 0, self.depth)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
        In order to write a good evaluation function we need to consider:
        - the current score, 
        - number of foods left and where are these foods, maybe closest food and furthest food? 
        - number of capsules left and where are these capsules, 
        - where are the ghosts,
        - where are the scared ghosts and how long they will remain scared 

        Score changes:
        - each food -> +10
        - end game -> +500 *the most important one
        - scared ghost -> +200
        - ghost -> -500 *This should not happen 
        - travel -> -1 
        So here our evaluation function must let pacman finish the game while try not to get cought by a ghost
        Also catching scared ghost is +200 points so, pacman should try to get a sacred ghost.
        I try to assign constant values to each of the variables that are important on the score and get a linear 
        equation for the evaluation function. First I used large numbers like 5000, 10000 but result 
        was not good enough. So then I used small numbers and results are better than previous. 
    """

    currentPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostPos = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostPos]
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    paraFood = 1
    paraFood2 = 1
    paraGhost = 1
    """
        Actually, I spent a lot of tiome on question 1. So basically I used the same strategy here but this time 
        with current state. Same parameter did not pass the autograder test so I added this constant parameters and 
        tried to tune them to get a better evaluation function.
    """
    if len(foodList) != 0:  # I added this because I got zero division error
        distFood = manhattanDistance(currentPos, min(foodList)) # Dist to closest food
        paraFood = 1 / distFood * 2
        distFood2 = manhattanDistance(currentPos, max(foodList)) # Dist to furthest food
        paraFood2 = 1 / distFood2
    distGhost = manhattanDistance(currentPos, min(currentGameState.getGhostPositions()))

    if distGhost < 2 and scaredTimes[0] < 3:  # distGhost is very sensitive, the best results are at 2
        paraGhost = -distGhost # I just use large value here because this is the most important parameter
    elif distGhost < 2 and scaredTimes[0] > 2:
        paraGhost = 1 / distGhost * 5


    return currentGameState.getScore() + paraGhost + paraFood + paraFood2


# Abbreviation
better = betterEvaluationFunction
