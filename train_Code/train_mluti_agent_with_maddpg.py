'''
使用maddpg训练多智能体环境
1、多智能体环境由Unty3D构建，每个智能体对应一个brain，并由UnityEnv使用gym封装。
2、具体的做法如下
    （1）实现每个智能体对应一个ddpg算法
    （2）每个ddpg算法，actor网络以智能体的观测值为输入，critic网络以全部智能体的观测值和动作为输入
    （3）实现模型训练的实时tensorboard监测，以及模型的加载
'''
import sys
sys.path.append('/Users/liyang/Code/GitHubCode/')
import RLs.Algorithms.ddpg
from UnityEnvTools import UnityEnv


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        summary_writer = tf.summary.FileWriter("./summary/" + "adversary_%d" % i)
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg'), summary_writer=summary_writer))
    for i in range(num_adversaries, env.n):
        summary_writer = tf.summary.FileWriter("./summary/" + "good_%d" % i)
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg'), summary_writer=summary_writer))
    return trainers


def train():
    pass


def write_summary(summary_writer, episode, all_agents_episode_reward):
        """
        Saves training statistics to Tensorboard.
        :param lesson_num: Current lesson number in curriculum.
        :param global_step: The number of steps the simulation has been going for
        """
        summary = tf.Summary()
        summary.Value.add(tag='all_agents_episode_reward', simple_value=all_agents_episode_reward)
        #summary.value.add(tag='all_agents_episode_reward', simple_value=all_agents_episode_reward)
        summary_writer.add_summary(summary, episode)
        summary_writer.flush()


if __name__ == "__main__":
    print(sys.path)