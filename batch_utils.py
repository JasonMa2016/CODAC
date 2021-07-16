import numpy as np
import torch
import wandb


def exploration_before_start(args, env_sampler, env_pool, agent, init_exploration_steps=5000):
    # init_exploration_steps = 5000
    for i in range(init_exploration_steps):
        state, action, next_state, reward, done, info = env_sampler.sample(agent, random_explore=True)
        env_pool.push(state, action, reward, next_state, done)


def evaluate_policy(args, env_sampler, agent, epoch_length=1000):
    env_sampler.current_state = None
    env_sampler.path_length = 0

    sum_reward = 0
    sum_cost = 0
    for t in range(epoch_length):
        state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)

        sum_reward += reward
        if done:
            break
    # reset the environment
    env_sampler.current_state = None
    env_sampler.path_length = 0

    return sum_reward


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0
    if train_step > args.max_train_repeat_per_step * cur_step:
        return 0

    # num_train_repeat: 20
    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size
        model_reward = np.array([0.])
        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))

            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)

        batch_mask = 1 - batch_done

        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
            (batch_state, batch_action, batch_reward, batch_next_state, batch_mask), args.policy_train_batch_size, i)

        if args.wandb:
            wandb.log({'Training/critic1_loss': critic_1_loss,
                        'Training/critic2_loss': critic_2_loss,
                        'Training/policy_loss': policy_loss,
                        'Training/entropy_loss': ent_loss,
                        'Training/alpha': alpha})

    return args.num_train_repeat
