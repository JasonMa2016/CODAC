from env.ant_circle import AntCircleEnv
from env.ant_obstacle import AntObstacleEnv
from env.humanoid_circle import HumanoidCircleEnv
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