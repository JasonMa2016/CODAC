import gym
from env.ant_obstacle import AntObstacleEnv
from gym.envs.registration import register

register(
    id='AntCircle-v0',
    entry_point='env:AntCircleEnv',
)

register(
    id='AntObstacle-v0',
    entry_point='env:AntObstacleEnv',
)

register(
    id='HumanoidCircle-v0',
    entry_point='env:HumanoidCircleEnv',
)