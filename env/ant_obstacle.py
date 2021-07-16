import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET

from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 20.0,
}

GYM_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),
    'assets')

class AntObstacleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file=os.path.join(GYM_ASSETS_DIR, 'ant_obstacle.xml'),
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())

        self.set_risk()

        self.target_goal = np.array([10., 10.])
        self.obstacle = np.array([5., 5.])
        self.r = 3

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def set_risk(self, risk_prob=0.9, risk_penalty=500):
        self.risk_prob = risk_prob
        self.risk_penalty = risk_penalty

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        velocity = np.linalg.norm(xy_velocity)

        d_goal = np.linalg.norm(self.target_goal - self.get_xy())
        reward = -d_goal + 0.1 * velocity
        unsafe = np.linalg.norm(self.obstacle - self.get_xy()) < self.r

        cost = 0
        if unsafe:
            cost = 1
            u = np.random.uniform(0, 1)
            if u > self.risk_prob:
                reward = reward - self.risk_penalty

        done = d_goal < 1
        if done:
            reward += 1
        observation = self._get_obs()
        info = {'cost': cost}

        return observation, reward, done, info

    def get_xy(self):
        return self.sim.data.qpos[:2]

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset(self, pos=None):
        self.sim.reset()
        ob = self.reset_model(pos=pos)
        return ob

    def reset_model(self, pos=None):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        sampled = False
        while not sampled:
            x_init, y_init = self.np_random.uniform(0, 7, size=(2,))
            d_circle = (x_init-5.0)**2 + (y_init-5.0)**2
            if d_circle > (self.r ** 2):
                sampled = True
            qpos[:2] = [x_init, y_init]
        if pos is not None:
            qpos[:2] = pos
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
        self.viewer.cam.lookat[0] += 4
        self.viewer.cam.lookat[1] += 4

