# UnityEnvTools

## 

###

UnityEnvTools be maked for Unity mlagents environment like gym.
It has three attributes:
- 1 observation_space  like {'RacerAdversaryLearnig': Box(13,), 'RacerTargetLearning': Box(13,)}
- 2  action_space like {'RacerAdversaryLearnig': Box(2,), 'RacerTargetLearning': Box(2,)}

and has methods:
- 1 obs = env.reset()
    output: {'RacerAdversaryLearnig':[[......],....],'RacerTargetLearning': [[....],....]}
- 2 obs, reward, done, info = env.step(action)
    input: {'RacerAdversaryLearnig': array([-0.28951204,  0.17042696], dtype=float32), 'RacerTargetLearning': array([0.1588485 , 0.21967688], dtype=float32)}
    output: {'RacerAdversaryLearnig':[[......],....],'RacerTargetLearning': [[....],....]}
            {'RacerAdversaryLearnig': [0.4701349437236786], 'RacerTargetLearning': [-2.040186643600464]}
            {'RacerAdversaryLearnig': [False], 'RacerTargetLearning': [False]}

