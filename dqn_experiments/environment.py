class GridWorld():
	"""
	Gridworld environment along with functions that allow an RL agent
	to navigate in the environment. Functions mimic OpenAI domain functions.
	"""

	def __init__(self, sentence):
		self.state = self.reset()


	def reset():
		"""
		Returns a random starting state.

		:returns state
		:rtype numpy array
		"""

		return self.state
		raise NotImplementedError


	def step(self, action):
		"""
		For a given action
    	returns:
    		- next state
    		- reward for the state, action pair
    		- whether or not the agent is now in a terminal state (use isTerminal function with next state as input)

    	:param action: action that the agent has taken just now

    	:returns: next state
    	:rtype: numpy array
    	:returns: reward
    	:rtype: float
    	:returns: whether or not episode is done
    	:rtype: boolean
    	"""

    	raise NotImplementedError


    def isTerminal(self, state, sentence):
    	"""
    	For a given image and instruction,
    	returns whether or not the agent is in
    	a terminal state.

    	:param state: a numpy array of the current image
    	:param sentence: a numpy array of the sentence instruction
    	:returns: whether or not current state is a terminal state
    	:rtype: boolean
    	"""

    	raise NotImplementedError

    def reward(self, action):
    	"""
    	For a given action, 
    	returns: the reward the agent gets for being in 
    	the current state and taking the selected action.

    	:
    	"""
    	raise NotImplementedError




