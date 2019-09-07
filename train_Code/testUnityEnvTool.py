from UnityEnv.UnityEnvTool import UnityEnv
#from envs import UnityEnvironment
import numpy as np

if __name__ == "__main__":
    # env = UnityEnv(environment_filename='/Users/liyang/Code/UnityEnv/walker', worker_id=0, no_graphics=False)
    #env = UnityEnvironment(file_name='/Users/liyang/Code/UnityEnv/walker')
    env = UnityEnv()
    print(env.brain_names)
    print(env.brains)
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    print(env.agents)
    print(env.n)
    print(env.observation_space[0].shape[0])
    print(env.action_space[0].dtype)
    # for i in range(10000):
    #     action = []
    #     for brain_name in env.brain_names:
    #         action.append(env.action_space[env.brain_names.index(brain_name)].sample())
    #     obs, reward, done, info = env.step(action)

    env.close()
    import os
    print(os.path)
  

