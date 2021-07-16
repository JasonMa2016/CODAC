import gym
import pygame
from gym import spaces
from gym.utils import seeding
import numpy as np

class PointMass(gym.Env):
    def __init__(self, N=1, risk_prob=0.7, risk_penalty=200):
        # Step 1: Car parameterss
        self.v_max = 0.1
        self.v_sigma = 0.01
        # Step 3: Environment parameters
        self.d_safe = 0.1
        self.d_goal = 0.05
        self.d_sampling = 0.1
        self.init_pos = np.array([1.0, 1.0])
        self.risk_prob = risk_prob
        self.risk_penalty = risk_penalty
        self.N = N # number of obstacles

        self.low_state = 0
        self.high_state= 1

        self.min_actions = np.array(
            [-self.v_max, -self.v_max], dtype=np.float32
        )
        self.max_actions = np.array(
            [self.v_max, self.v_max], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2+2, ),
            dtype=np.float32
        )

        self.goal = np.array([0.1, 0.1])
        self.r = 0.3 # obstacle radius
        self.centers = np.array([0.5, 0.5])

        # Step 4: Rendering parameters
        self.screen_size = [600, 600]
        self.screen_scale = 600
        self.background_color = [255, 255, 255]
        self.wall_color = [0, 0, 0]
        self.circle_color = [255, 0, 0]
        self.safe_circle_color = [200,0,0]
        self.lidar_color = [0, 0, 255]
        self.goal_color = [0, 255, 0]
        self.robot_color = [0, 0, 0]
        self.safety_color = [255, 0, 0]
        self.goal_size = 15
        self.radius = 9
        self.width = 3
        self.pygame_init = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

    def reset(self, eval=False):

        sampled = False
        while not sampled:
            # uniform state space initial state distribution
            self.init_pos = self.np_random.uniform(0.1, 0.9, size=(2,))

            if self.is_safe(self.init_pos):
                sampled = True
        if eval:
            self.init_pos = np.array([0.85, 0.85]) + self.np_random.uniform(-0.05, 0, size=(2,))

        self.state = np.array(list(self.init_pos) + list(self.goal))
        return np.array(self.state)


    def get_dist_to_goal(self, state):
        return np.linalg.norm(state[-2:]-state[:2])

    # Check if the state is safe.
    def is_safe(self, state):
        if len(state.shape) == 1:
            safe = True
            d_circle = (state[0]-self.centers[0])**2 + (state[1]-self.centers[1])**2
            if d_circle <= (self.r ** 2):
                safe = False
            return safe
        elif len(state.shape) == 2:
            d_circle = (state[:, 0] - 0.5) ** 2 + (state[:, 1] - 0.5) ** 2
            safe = (d_circle > self.r ** 2).astype(float)
            return safe

    def step(self, action):
        action = np.clip(action, -self.v_max, self.v_max)
        assert self.action_space.contains(action)

        d_goal = self.get_dist_to_goal(self.state)
        reward = - d_goal - 0.1

        cost = 0
        if not self.is_safe(self.state):
            u = np.random.uniform(0, 1)
            if u > self.risk_prob:
                cost = 1
                reward -= self.risk_penalty
        done = 0
        if d_goal < self.d_goal:
            done = 1

        self.state[:2] = self.state[:2] + action
        self.state = np.clip(self.state, self.low_state, self.high_state)

        return np.array(self.state), reward, done, {'cost':cost}

    def render(self):
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(self.background_color)

        p_car = self.state[:2]

        p = (self.screen_scale * p_car).astype(int).tolist()
        pygame.draw.circle(self.screen, self.robot_color, p, self.radius, self.width)

        c, r = (self.screen_scale*self.centers[:2]).astype(int), int(self.screen_scale*self.r)
        pygame.draw.circle(self.screen, self.circle_color, c, r)

        pygame.draw.circle(self.screen, self.goal_color, (self.screen_scale * self.goal).astype(int), self.goal_size)
        pygame.display.flip()

        self.clock.tick(20)