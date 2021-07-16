import numpy as np
import torch


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))

    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)

    reward = np.reshape(reward, (reward.shape[0], -1))
    labels = np.concatenate((reward, delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256)

    torch.save(predict_env.model.state_dict(), f'saved_models/{args.env}-ensemble.pt')


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    # rollout_batch_size = 50000
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)

    for i in range(rollout_length):
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action, reward_penalty=args.penalty,
                                                                algo=args.algo)
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, eval_t=False, random_explore=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        if not random_explore:
            action = agent.select_action(self.current_state, eval_t)
        else:
            action = self.env.action_space.sample()

        next_state, reward, terminal, info = self.env.step(action)
        # if eval_t:
        #     self.env.render()
        self.path_length += 1
        self.sum_reward += reward

        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info