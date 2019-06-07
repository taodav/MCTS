import numpy as np
import copy

class GridWorld:
    VALIDATION_MODE = 0

    def __init__(self, **kwargs):
        self._size_maze = 8
        self._higher_dim_obs = kwargs.get("higher_dim_obs", False)
        self.create_map()
        self.intern_dim = 2

    def create_map(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        self._map[-1, :] = 1
        self._map[0, :] = 1
        self._map[:, 0] = 1
        self._map[:, -1] = 1
        self._map[:, self._size_maze // 2] = 1
        self._map[self._size_maze // 2, self._size_maze // 2] = 0
        self._pos_agent = [2, 2]
        self._pos_goal = [self._size_maze - 2, self._size_maze - 2]

    def reset(self):
        self.create_map()

        self._map[self._size_maze // 2, self._size_maze // 2] = 0

        # Setting the starting position of the agent
        self._pos_agent = [self._size_maze // 2, self._size_maze // 2]


        return [1 * [self._size_maze * [self._size_maze * [0]]]]

    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier 
            included between 0 included and nActions() excluded.
        """

        self._cur_action = action
        if (action == 0):
            if (self._map[self._pos_agent[0] - 1, self._pos_agent[1]] == 0):
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif (action == 1):
            if (self._map[self._pos_agent[0] + 1, self._pos_agent[1]] == 0):
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif (action == 2):
            if (self._map[self._pos_agent[0], self._pos_agent[1] - 1] == 0):
                self._pos_agent[1] = self._pos_agent[1] - 1
        elif (action == 3):
            if (self._map[self._pos_agent[0], self._pos_agent[1] + 1] == 0):
                self._pos_agent[1] = self._pos_agent[1] + 1

        # There is no reward in this simple environment
        self.reward = 0

        if self._pos_agent == self._pos_goal:
            self.reward = 1

        return self.reward

    def inputDimensions(self):
        if (self._higher_dim_obs == True):
            return [(1, self._size_maze * 6, self._size_maze * 6)]
        else:
            return [(1, self._size_maze, self._size_maze)]

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return 4

    def observe(self):
        obs = copy.deepcopy(self._map)

        obs[self._pos_agent[0], self._pos_agent[1]] = 0.5
        if (self._higher_dim_obs == True):
            "self._pos_agent"
            self._pos_agent
            obs = self.get_higher_dim_obs([self._pos_agent], [self._pos_goal])

        return [obs]

    def get_higher_dim_obs(self, indices_agent, indices_reward):
        """ Obtain the high-dimensional observation from indices of the agent position and the indices of the reward positions.
        """
        obs = copy.deepcopy(self._map)
        obs = obs / 1.
        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)
        # agent repr
        agent_obs = np.zeros((6, 6))
        agent_obs[0, 2] = 0.7
        agent_obs[1, 0:5] = 0.8
        agent_obs[2, 1:4] = 0.8
        agent_obs[3, 1:4] = 0.8
        agent_obs[4, 1] = 0.8
        agent_obs[4, 3] = 0.8
        agent_obs[5, 0:2] = 0.8
        agent_obs[5, 3:5] = 0.8

        # reward repr
        reward_obs = np.zeros((6, 6))
        # reward_obs[:,1]=0.8
        # reward_obs[0,1:4]=0.7
        # reward_obs[1,3]=0.8
        # reward_obs[2,1:4]=0.7
        # reward_obs[4,2]=0.8
        # reward_obs[5,2:4]=0.8

        for i in indices_reward:
            obs[i[0] * 6:(i[0] + 1) * 6:, i[1] * 6:(i[1] + 1) * 6] = reward_obs

        for i in indices_agent:
            obs[i[0] * 6:(i[0] + 1) * 6:, i[1] * 6:(i[1] + 1) * 6] = agent_obs

        # plt.imshow(obs, cmap='gray_r')
        # plt.show()
        return obs

    def inTerminalState(self):
        # Uncomment the following lines to add some cases where the episode terminates.
        # This is used to show how the environment representation interpret cases where 
        # part of the environment could not be explored.
        #        if((self._pos_agent[0]<=1 and self._cur_action==0) ):
        #            return True

        # If there is a goal, then terminates the environment when the goas is reached.
        return self._pos_agent == self._pos_goal


if __name__ == "__main__":
    env = GridWorld()
    env1 = copy.deepcopy(env)
    print("here")
