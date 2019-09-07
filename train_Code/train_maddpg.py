"""
使用maddpg训练多智能体环境
1、多智能体环境由Unty3D构建，每个智能体对应一个brain，并由UnityEnv使用gym封装。
2、具体的做法如下
    （1）实现每个智能体对应一个ddpg算法
    （2）每个ddpg算法，actor网络以智能体的观测值为输入，critic网络以全部智能体的观测值和动作为输入
    （3）实现模型训练的实时tensorboard监测，以及模型的加载
"""
# import sys
# sys.path.append('/Users/liyang/Code/GitHubCode/')
import tensorflow as tf
from RLs.Algorithms.ddpg import DDPG
from UnityEnvTools.UnityEnv.UnityEnvTool import UnityEnv
from RLs.Algorithms import config
from RLs.config import train_config
import os
from loop import Loop


def train():
    env = UnityEnv()
    action_spaces = env.action_space
    observation_spaces = env.observation_space
    algorithm_config = config.ddpg_config
    # 实现每个智能体对应一个ddpg算法
    models = [DDPG(
        s_dim=observation_spaces[i].shape[0],
        visual_sources=0,
        visual_resolutions=0,
        a_dim_or_list=action_spaces[i].shape[0],
        action_type=env.action_type,
        cp_dir='./' + 'agent' + str(i) + '/' + 'model',
        log_dir='./' + 'agent' + str(i) + '/' + 'log',
        excel_dir='./' + 'agent' + str(i) + '/' + 'excel',
        logger2file=False,
        out_graph=True,
        **algorithm_config
    ) for i in range(env.n)]


    params = {
                'env': env,
                'brain_names': brain_names,
                'models': models,
                'begin_episode': begin_episode,
                'save_frequency': save_frequency,
                'reset_config': reset_config,
                'max_step': max_step,
                'max_episode': max_episode,
                'sampler_manager': sampler_manager,
                'resampling_interval': resampling_interval
            }

    Loop.train_perStep(**params)


def write_summary(summary_writer, episode, all_agents_episode_reward):
    """
        Saves training statistics to Tensorboard.
        :param lesson_num: Current lesson number in curriculum.
        :param global_step: The number of steps the simulation has been going for
        """
    summary = tf.Summary()
    summary.Value.add(tag='all_agents_episode_reward', simple_value=all_agents_episode_reward)
    # summary.value.add(tag='all_agents_episode_reward', simple_value=all_agents_episode_reward)
    summary_writer.add_summary(summary, episode)
    summary_writer.flush()


if __name__ == "__main__":
    pass
