from UnityEnv import UnityEnv
from mlagents.envs import UnityEnvironment

if __name__ == "__main__":
    env = UnityEnvironment(file_name='/Users/liyang/Code/UnityEnv/Soccer', worker_id=0, no_graphics=True)
    # #env = UnityEnvironment(file_name='UnityEnvTools/soccer/soccer', worker_id=1, no_graphics=True)
    # print(env.action_space)
    # print(env.observation_space)
    # print(env.brain_names)
    # print(env.brains)
    # print(env.observation_space)
    # print(env.action_space)
    # obs = env.reset()
    # print(obs)
    # env.close()
    info = env.reset()
    env.close