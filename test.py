from UnityEnv import UnityEnv
from mlagents.envs import UnityEnvironment
import numpy as np

if __name__ == "__main__":
    env = UnityEnv(environment_filename='/Users/liyang/Code/UnityEnv/ship', worker_id=0, no_graphics=False)
    #env = UnityEnvironment(file_name='/Users/liyang/Code/UnityEnv/walker')
    print(env.brain_names)
    print(env.brains)
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    for i in range(1):
        action = {}
        for brain_name in env.brain_names:
            action[brain_name] = env.action_space[brain_name].sample()
        obs, reward, done, info = env.step(action)
        
    # info = env.reset()[env.brain_names[0]]
    # obs = info.vector_observations
    # print(np.array(obs).shape)
    # print(obs)
    # count = 0
    # for i in obs[0]:
    #     if i == 0.0:
    #         count += 1
    # print(count)
    env.close()
  