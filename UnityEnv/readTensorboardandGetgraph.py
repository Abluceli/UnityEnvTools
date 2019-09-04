from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 
# 遍历指定目录，显示目录下的所有文件名
def getTensorboardFiles(filepath):
    pathDir =  os.listdir(filepath)
    #print(pathDir)
    tensorboardFiles = []
    for allDir in pathDir:
        child = os.path.join('%s%s%s' % (filepath, '\\', allDir))
        tensorboardFiles.append(child)
    return tensorboardFiles

def readTensorboardFile(fileName):
    reward_info = [[],[]]
    ea = event_accumulator.EventAccumulator(fileName)
    ea.Reload()
    # ['Policy/Entropy', 'Policy/Learning Rate', 'Policy/Extrinsic Value Estimate', 'Environment/Lesson', 
    # 'Environment/Cumulative Reward', 'Environment/Episode Length', 'Policy/Extrinsic Reward', 'Losses/Value Loss',
    #  'Losses/Policy Loss']
    #print(ea.scalars.Keys())

    # for key in ea.scalars.Keys():
    #     print(len(ea.scalars.Items(key)))
    for info in ea.scalars.Items('Environment/Cumulative Reward'):
        reward_info[0].append(info.step)
        reward_info[1].append(info.value)
    return reward_info

def drawTensorboardInfo(tensorboardInfo):
    plt.figure()
    sns.set_style('darkgrid')
    sns.set_context('paper')
    plt.plot(tensorboardInfo[0], tensorboardInfo[1])
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Cumulative Reward')
    plt.show()
    plt.close()

if __name__ == "__main__":
    filePaths = [r'C:\Users\Abluc\Desktop\ml-agents-master-v9\summaries\Predator-prey_with_landmark_shape_reward_Adversary01Learning',
                 r'C:\Users\Abluc\Desktop\ml-agents-master-v9\summaries\Predator-prey_with_landmark_shape_reward_Adversary02Learning',
                 r'C:\Users\Abluc\Desktop\ml-agents-master-v9\summaries\Predator-prey_with_landmark_shape_reward_Good01Learning']
    draw_infos = [[],[]]
    for filePath in filePaths:
        tensorboardFiles = getTensorboardFiles(filePath)
        draw_info = [[],[]]
        for files in tensorboardFiles:
            reward_info = readTensorboardFile(files)
            draw_info[0] = np.concatenate((draw_info[0], reward_info[0]))
            draw_info[1] = np.concatenate((draw_info[1], reward_info[1]))
        # print(draw_info[0], draw_info[1])
        drawTensorboardInfo(draw_info)      
        # draw_infos.append(draw_info)
        if len(draw_infos[1]) == 0:
            draw_infos[1] = draw_info[1]
        else:
            draw_infos[1] = np.array(draw_infos[1]) + np.array(draw_info[1])
    drawTensorboardInfo(draw_infos)
    # for info in draw_infos:
    #     print(len(info))
