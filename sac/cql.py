import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy


class CQL(object):
    def __init__(self, num_inputs, action_space,
                 ## SAC
                 gamma=0.99, tau=0.005, alpha=0.2,
                 policy='Gaussian',
                 target_update_interval=1,
                 automatic_entropy_tuning=True,
                 target_entropy=-3,
                 hidden_size=256,
                 lr=0.0003,

                 ## CQL
                 min_q_version=3,
                 temp=1.0,
                 min_q_weight=5.0,
                 max_q_backup=False,
                 deterministic_backup=False,
                 num_random=10,
                 with_lagrange=True,
                 lagrange_thresh=10.0,
                 policy_lr=3e-5,
                 policy_eval_start=0,
                 device='cuda'):

        ## SAC
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.policy_eval_start = policy_eval_start
        self.updates = 0

        ## CQL
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        self.device = device

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                if self.target_entropy != 'auto':
                    self.target_entropy = target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=policy_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_prime_optimizer = Adam([self.log_alpha_prime], lr=lr)

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        pred1, pred2 = network(obs_temp, actions)
        pred1 = pred1.view(obs.shape[0], num_repeat, 1)
        pred2 = pred2.view(obs.shape[0], num_repeat, 1)

        return pred1, pred2

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, new_obs_log_pi, _ = network.sample(obs_temp)
        return new_obs_actions.detach(), new_obs_log_pi.view(obs.shape[0], num_actions, 1).detach()

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        obs = torch.FloatTensor(state_batch).to(self.device)
        next_obs = torch.FloatTensor(next_state_batch).to(self.device)
        actions= torch.FloatTensor(action_batch).to(self.device)
        rewards = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        masks = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_pi, _ = self.policy.sample(obs)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        q_new_actions = torch.min(*self.critic(obs, new_obs_actions))
        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        if self.updates < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k
            gradient steps here, or not having it
            """
            self.updates += 1
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (self.alpha * log_pi - policy_log_prob).mean()

        """
        QF Loss
        """
        q1_pred, q2_pred = self.critic(obs, actions)
        new_next_actions, new_log_pi, _ = self.policy.sample(next_obs)

        new_curr_actions, new_curr_log_pi, _ = self.policy.sample(obs)

        if not self.max_q_backup:
            target_qf1, target_qf2 = self.critic_target(next_obs, new_next_actions)
            target_q_values = torch.min(target_qf1, target_qf2)
            if not self.deterministic_backup:
                target_q_values = target_q_values - self.alpha * new_log_pi
        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values, target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.critic_target)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = rewards + masks * self.gamma * target_q_values
        q_target = q_target.detach()

        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        ## CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1,1).to(self.device)

        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random,
                                                                     network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random,
                                                                        network=self.policy)
        q1_rand, q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic)
        q1_curr_actions, q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic)
        q1_next_actions, q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic)

        # when is this ever used?
        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        # std_q1 = torch.std(cat_q1, dim=1)
        # std_q2 = torch.std(cat_q2, dim=1)

        if self.min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(),
                 q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(),
                 q2_curr_actions - curr_log_pis.detach()], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        qf_loss = qf1_loss + qf2_loss


        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_optim.step()


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, path):
        if not os.path.exists('saved_policies/'):
            os.makedirs('saved_policies/')

        actor_path = path+'-actor.pt'
        critic_path = path + '-critic.pt'
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, path):
        actor_path = path+'-actor.pt'
        critic_path = path + '-critic.pt'
        self.policy.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
