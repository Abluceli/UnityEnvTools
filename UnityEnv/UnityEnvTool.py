import logging
import itertools
import gym
import numpy as np
from mlagents.envs import UnityEnvironment
from gym import error, spaces

class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")

""""
There are four type mlagents env in the Unityenv:
1. one brain one agent
2. one brain multi agents
3. multi brain one agent of each brain 
4. multi brain multi agents of each brain
"""
class UnityEnv(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    Multi-agent environments use lists for object types, as done here:
    https://github.com/openai/multiagent-particle-envs
    """

    def __init__(
        self,
        environment_filename: str = None,
        worker_id: int = 0,
        no_graphics: bool = False,
        multibrain: bool = False,
        multiagent: bool = False,
        use_visual: bool = False,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_visual_obs: bool = False,
    ):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment.
        :param use_visual: Whether to use visual observation or vector observation.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param multiagent: Whether to run in multi-agent mode (lists of obs, reward, done).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param no_graphics: Whether to run the Unity simulator in no-graphics mode
        :param allow_multiple_visual_obs: If True, return a list of visual observations instead of only one.
        """
        if environment_filename == None:
            self._env = UnityEnvironment(worker_id=worker_id, no_graphics=no_graphics)
        else:
            self._env = UnityEnvironment(
                file_name=environment_filename, worker_id=worker_id, no_graphics=no_graphics
            )
        self.name = self._env.academy_name
        self.brains = self._env.brains
        self.brain_names = self._env.brain_names
        self.action_space = []
        self.observation_space = []
        self.n = len(self.brains)
        # Set observation and action spaces
        for brain_name in self.brain_names:
            brain = self.brains[brain_name]
            if brain.vector_action_space_type == "discrete":
                self.action_type = 'discrete'
                if len(brain.vector_action_space_size) == 1:
                    self.action_space.append(spaces.Discrete(brain.vector_action_space_size[0]))
                else:
                    # if flatten_branched:
                    #     self._flattener = ActionFlattener(brain.vector_action_space_size)
                    #     self._action_space = self._flattener.action_space
                    # else:
                    self.action_space.append(spaces.MultiDiscrete(
                        brain.vector_action_space_size
                    ))

            else:
                self.action_type = 'continuous'
                high = np.array([1] * brain.vector_action_space_size[0])
                self.action_space.append(spaces.Box(-high, high, dtype=np.float32))
            #np.inf mean the max number in python
            high = np.array([np.inf] * brain.vector_observation_space_size * brain.num_stacked_vector_observations)
            #self.action_meanings = brain.vector_action_descriptions
            if use_visual:
                if brain.camera_resolutions[0]["blackAndWhite"]:
                    depth = 1
                else:
                    depth = 3
                self._observation_space = spaces.Box(
                    0,
                    1,
                    dtype=np.float32,
                    shape=(
                        brain.camera_resolutions[0]["height"],
                        brain.camera_resolutions[0]["width"],
                        depth,
                    ),
                )
            else:
                self.observation_space.append(spaces.Box(-high, high, dtype=np.float32))

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
       
        info_all = self._env.reset()
        self.agents = {}
        obs = []
        for brain_name in self.brain_names:
            brain_info = info_all[brain_name]
            obs.append(brain_info.vector_observations[0])
            self.agents[brain_name] = len(brain_info.vector_observations)
        return obs

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        In the case of multi-agent environments, these are lists.
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """

        # Use random actions for all other agents in environment.
        # if self._multiagent:
        #     if not isinstance(action, list):
        #         raise UnityGymException(
        #             "The environment was expecting `action` to be a list."
        #         )
        #     if len(action) != self._n_agents:
        #         raise UnityGymException(
        #             "The environment was expecting a list of {} actions.".format(
        #                 self._n_agents
        #             )
        #         )
        #     else:
        #         if self._flattener is not None:
        #             # Action space is discrete and flattened - we expect a list of scalars
        #             action = [self._flattener.lookup_action(_act) for _act in action]
        #         action = np.array(action)
        # else:
        #     if self._flattener is not None:
        #         # Translate action into list
        #         action = self._flattener.lookup_action(action)
        brains_action = {}
        for brain in self.brain_names:
            brains_action[brain] = action[self.brain_names.index(brain)]
        info_all = self._env.step(brains_action)
        obs = []
        reward = []
        done = []
        
        for brain_name in self.brain_names:
            brain_info = info_all[brain_name]
            obs.append(brain_info.vector_observations[0])
            reward.append(brain_info.rewards[0])
            done.append(brain_info.local_done[0])
        # n_agents = len(info.agents)
        # self._check_agents(n_agents)
        # self._current_state = info

        # if not self._multiagent:
        #     obs, reward, done, info = self._single_step(info)
        #     self.game_over = done
        # else:
        #     obs, reward, done, info = self._multi_step(info)
        #     self.game_over = all(done)
        return obs, reward, done, "info"


#     def _preprocess_single(self, single_visual_obs):
#         if self.uint8_visual:
#             return (255.0 * single_visual_obs).astype(np.uint8)
#         else:
#             return single_visual_obs

#     def _preprocess_multi(self, multiple_visual_obs):
#         if self.uint8_visual:
#             return [
#                 (255.0 * _visual_obs).astype(np.uint8)
#                 for _visual_obs in multiple_visual_obs
#             ]
#         else:
#             return multiple_visual_obs

    def render(self, mode="rgb_array"):
        pass

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()



    def seed(self, seed=None):
        pass

#     def _check_agents(self, n_agents):
#         if not self._multiagent and n_agents > 1:
#             raise UnityGymException(
#                 "The environment was launched as a single-agent environment, however"
#                 "there is more than one agent in the scene."
#             )
#         elif self._multiagent and n_agents <= 1:
#             raise UnityGymException(
#                 "The environment was launched as a mutli-agent environment, however"
#                 "there is only one agent in the scene."
#             )
#         if self._n_agents is None:
#             self._n_agents = n_agents
#             logger.info("{} agents within environment.".format(n_agents))
#         elif self._n_agents != n_agents:
#             raise UnityGymException(
#                 "The number of agents in the environment has changed since "
#                 "initialization. This is not supported."
#             )

    # @property
    # def metadata(self):
    #     return {"render.modes": ["rgb_array"]}

    # @property
    # def reward_range(self):
    #     return -float("inf"), float("inf")

    # @property
    # def spec(self):
    #     return None

    # @property
    # def action_space(self):
    #     return self.action_space

    # @property
    # def observation_space(self):
    #     return self.observation_space

    # @property
    # def number_agents(self):
    #     return self.agents


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """
    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]
