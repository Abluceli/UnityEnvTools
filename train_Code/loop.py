import numpy as np

def get_visual_input(n, cameras, brain_obs):
    '''
    inputs:
        n: agents number
        cameras: camera number
        brain_obs: observations of specified brain, include visual and vector observation.
    output:
        [vector_information, [visual_info0, visual_info1, visual_info2, ...]]
    '''
    ss = []
    for j in range(n):
        s = []
        for k in range(cameras):
            s.append(brain_obs.visual_observations[k][j])
        ss.append(np.array(s))
    return ss

class Loop(object):
    def train_perStep(self, env, brain_names, models, begin_episode, save_frequency, reset_config, max_step, max_episode, sampler_manager, resampling_interval):
        """
        usually off-policy algorithms with replay buffer, i.e. dqn, ddpg, td3, sac
        also used for some on-policy algorithms, i.e. ac, a2c
        """
        agent_num = env.n
        state = [0] * agent_num
        visual_state = [0] * agent_num
        action = [0] * agent_num
        dones_flag = [0] * agent_num
        agents_num = [0] * agent_num
        rewards = [0] * agent_num

        for episode in range(begin_episode, max_episode):
            # if episode % resampling_interval == 0:
            #     reset_config.update(sampler_manager.sample_all())
            #obs = env.reset(config=reset_config, train_mode=True)
            obs = env.reset()
            for i, ob in enumerate(obs):
                agents_num[i] = 1
                dones_flag[i] = np.zeros(agents_num[i])
                rewards[i] = np.zeros(agents_num[i])
                state[i] = ob
                visual_state[i] = [[]]
            step = 0
            while True:
                step += 1
                for i, brain_name in enumerate(brain_names):
                    action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)
                for i, brain_name in enumerate(brain_names):
                    dones_flag[i] += obs[brain_name].local_done
                    next_state = obs[brain_name].vector_observations
                    next_visual_state = get_visual_input(agents_num[i], models[i].visual_sources, obs[brain_name])
                    models[i].store_data(
                        s=state[i],
                        visual_s=visual_state[i],
                        a=action[i],
                        r=np.array(obs[brain_name].rewards),
                        s_=next_state,
                        visual_s_=next_visual_state,
                        done=np.array(obs[brain_name].local_done)
                    )
                    state[i] = next_state
                    visual_state[i] = next_visual_state
                    models[i].learn(episode)
                    rewards[i] += np.array(obs[brain_name].rewards)
                if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                    break
            print(f'episode {episode} step {step}')
            for i in range(brains_num):
                models[i].writer_summary(
                    episode,
                    total_reward=rewards[i].mean(),
                    step=step
                )
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)
