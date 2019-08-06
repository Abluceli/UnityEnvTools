from UnityEnv import UnityEnv
from mlagents.envs import UnityEnvironment

if __name__ == "__main__":
    env = UnityEnv(environment_filename='/Users/liyang/Code/UnityEnv/ship', worker_id=0, no_graphics=False)

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
        print(obs,reward,done)
    env.close()
  