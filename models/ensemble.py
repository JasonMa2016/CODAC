import numpy as np
import gym
import itertools

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import wandb

device = torch.device('cuda')

def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    val = torch.fmod(torch.randn(size),2) * std
    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b


class ProbEnsemble(nn.Module):

    def __init__(self, state_size, action_size,
                 network_size=7, elite_size=5,
                 reward_size=1, hidden_size=200, lr=0.001):
        super().__init__()
        self.network_size = network_size
        self.num_nets = network_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_size = elite_size
        self.elite_model_idxes = []

        self.in_features = state_size + action_size
        self.out_features = 2 * (state_size + reward_size)

        self.lin0_w, self.lin0_b = get_affine_params(network_size, self.in_features, hidden_size)

        self.lin1_w, self.lin1_b = get_affine_params(network_size, hidden_size, hidden_size)

        self.lin2_w, self.lin2_b = get_affine_params(network_size, hidden_size, hidden_size)

        self.lin3_w, self.lin3_b = get_affine_params(network_size, hidden_size, hidden_size)

        self.lin4_w, self.lin4_b = get_affine_params(network_size, hidden_size, self.out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, self.out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, self.out_features // 2, dtype=torch.float32) * 10.0)
        self.fit_input = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def compute_decays(self):

        lin0_decays = 0.000025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.000075 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.000075 * (self.lin3_w ** 2).sum() / 2.0
        lin4_decays = 0.0001 * (self.lin4_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays + lin4_decays

    def fit_input_stats(self, data, device='cuda'):
        self.fit_input = True
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(device).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(device).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        if self.fit_input:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin4_w) + self.lin4_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        updated_count = 0
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / abs(best)
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True
                updated_count += 1
                improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def compute_loss(self, input, target):
        train_loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        train_loss += self.compute_decays()
        mean, logvar = self(input, ret_logvar=True)

        inv_var = torch.exp(-logvar)
        train_losses = ((mean - target) ** 2) * inv_var + logvar
        train_losses = train_losses.mean(-1).mean(-1)
        train_loss += train_losses.sum()

        return train_loss

    def train(self, inputs, targets, batch_size=256, holdout_ratio=0.2,
              max_logging=5000, max_epochs_since_update=5, max_epochs=None):

        self._max_epochs_since_update = max_epochs_since_update
        self._snapshots = {i: (None, 1e10) for i in range(self.num_nets)}
        self._epochs_since_update = 0

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        self.fit_input_stats(inputs)

        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        input_val = torch.from_numpy(holdout_inputs).float().to(device)
        target_val = torch.from_numpy(holdout_targets).float().to(device)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])

        if max_epochs is not None:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        grad_update = 0
        for epoch in epoch_iter:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]

                input = torch.from_numpy(inputs[batch_idxs]).float().to(device)
                target = torch.from_numpy(targets[batch_idxs]).float().to(device)
                train_loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                train_loss += self.compute_decays()
                mean, logvar = self(input, ret_logvar=True)

                inv_var = torch.exp(-logvar)
                train_losses = ((mean - target) ** 2) * inv_var + logvar
                train_losses = train_losses.mean(-1).mean(-1)
                train_loss += train_losses.sum()

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                grad_update += 1

            idxs = shuffle_rows(idxs)
            with torch.no_grad():
                ensemble_mean, ensemble_var = self(input_val, ret_logvar=False)
                ensemble_std = torch.sqrt(ensemble_var)

                rmse = torch.sqrt(((ensemble_mean - target_val.unsqueeze(0)) ** 2).mean(axis=[2, 1]))
                ll = torch.distributions.Normal(loc=ensemble_mean, scale=ensemble_std).log_prob(target_val)
                val_losses = -ll.mean(axis=[2, 1])

                # val_loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                # val_loss += self.compute_decays()
                # mean, logvar = self(input_val, ret_logvar=True)
                # inv_var = torch.exp(-logvar)
                # val_losses = ((mean - target_val) ** 2) * inv_var + logvar
                # val_losses = val_losses.mean(-1).mean(-1)
                #
                # val_loss += val_losses.sum()

                break_train = self._save_best(epoch, val_losses)

            if break_train:
                break
            # print(f"Epoch {epoch} Val {val_losses.mean().item()}")

        losses = val_losses.detach().cpu().numpy()
        sorted_loss_idx = np.argsort(losses)
        self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

        return rmse.mean().detach().cpu().item(), val_losses.mean().detach().cpu().item()

    def predict(self, inputs, batch_size=50000):
        ensemble_mean = np.zeros((self.network_size, inputs.shape[0], self.state_size + self.reward_size))
        ensemble_var = np.zeros((self.network_size, inputs.shape[0], self.state_size + self.reward_size))
        with torch.no_grad():
            for i in range(0, inputs.shape[0], batch_size):
                input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)

                pred_2d_mean, pred_2d_var = self(input, ret_logvar=False)
                ensemble_mean[:, i:min(i + batch_size, inputs.shape[0]), :] = pred_2d_mean.detach().cpu().numpy()
                ensemble_var[:, i:min(i + batch_size, inputs.shape[0]), :] = pred_2d_var.detach().cpu().numpy()

        return ensemble_mean, ensemble_var

    def compute_log_prob(self, input, target):
        ensemble_mu, ensemble_var = self(input.to(device))

        dist = torch.distributions.Normal(loc=ensemble_mu, scale=ensemble_var.sqrt())
        log_prob = dist.log_prob(target.to(device))
        log_prob = log_prob.mean(dim=-1, keepdim=True).mean(dim=0)

        return log_prob
