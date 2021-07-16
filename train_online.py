import argparse
import os
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

import wandb
from sac import SAC, CQL, ReplayMemory
from models import ProbEnsemble, PredictEnv

from batch_utils import *
from mbrl_utils import *
from utils import *

from tqdm import tqdm


def readParser():
    parser = argparse.ArgumentParser(description='BATCH_RL')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--env', default="pointmass",
                        help='Gym environment (default: pointmass)')
    parser.add_argument('--algo', default="sac")
    parser.add_argument('--adapt', dest='adapt', action='store_true')
    parser.set_defaults(adapt=False)
    parser.add_argument('--entropy', default="false")

    # risk parameters for the policy
    parser.add_argument('--risk_type', default="neutral")
    parser.add_argument('--risk_param', default=0.1)
    # risk parameters for the environment
    parser.add_argument('--risk_prob', type=float, default=0.8)
    parser.add_argument('--risk_penalty', type=float, default=200)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--dist_penalty_type', default="none")
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='reward penalty')

    parser.add_argument('--rollout_length', type=int, default=5, metavar='A',
                        help='rollout length')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--model_retain_epochs', type=int, default=5, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=1000, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=50000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--num_epoch', type=int, default=500, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='initial random exploration steps')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=1, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--eval_n_episodes', type=int, default=100, metavar='A',
                        help='number of evaluation episodes')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')
    parser.add_argument('--pre_trained', type=bool, default=False,
                        help='flag for whether dynamics model pre-trained')
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    total_step = 0
    reward_sum = 0
    rollout_length = args.rollout_length
    state_size = np.prod(env_sampler.env.observation_space.shape)
    qvel_size = int((state_size + 1) / 2)

    exploration_before_start(args, env_sampler, env_pool, agent, init_exploration_steps=1000)
    save_interval = int(args.num_epoch / 10)
    eval_interval = int(args.num_epoch / 100)

    for epoch_step in tqdm(range(args.num_epoch)):
        # save buffer for offline learning
        if (epoch_step+1) % save_interval == 0:
            buffer_path = f'dataset/{args.env}/{args.run_name}-epoch{epoch_step+1}.npy'
            env_pool.save_buffer(buffer_path)
            agent_path = f'saved_policies/{args.env}/online/{args.run_name}-epoch{epoch_step+1}'
            agent.save_model(agent_path)

        start_step = total_step
        train_policy_steps = 0
        env_sampler.current_state = None
        env_sampler.path_length = 0
        for i in range(args.epoch_length):
            cur_step = total_step - start_step

            if cur_step >= args.epoch_length:
                break

            # step in real environment
            state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(state, action, reward, next_state, done)

            # train policy
            if len(env_pool) > 1000:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
            total_step += 1

        if epoch_step % eval_interval == 0:
            rewards = [evaluate_policy(args, env_sampler, agent, args.epoch_length) for _ in range(args.eval_n_episodes)]
            rewards = np.array(rewards)
            rewards_avg = np.mean(rewards, axis=0)
            rewards_std = np.std(rewards, axis=0)
            sorted_rewards = np.sort(rewards)
            cvar = sorted_rewards[:int(0.1*sorted_rewards.shape[0])].mean()
            print("")
            print(f'Epoch {epoch_step} Eval_Reward {rewards_avg:.2f} Eval_Cvar {cvar:.2f} Eval_Std {rewards_std:.2f}')
            if args.wandb:
                wandb.log({'epoch':epoch_step,
                           'eval_reward': rewards_avg,
                           'eval_cvar0.1': cvar,
                           'reward_std': rewards_std})


def main():
    args = readParser()

    if args.env == "riskymass" or args.env == 'AntObstacle-v0':
        if args.algo == 'codac':
            run_name = f"online-{args.risk_prob}-{args.risk_penalty}-{args.algo}-{args.risk_type}{args.risk_param}-E{args.entropy}-{args.seed}"
            if args.adapt:
                run_name = f"online-{args.risk_prob}-{args.risk_penalty}-adapt-{args.algo}-{args.risk_type}{args.risk_param}-E{args.entropy}-{args.seed}"

        else:
            run_name = f"online-{args.risk_prob}-{args.risk_penalty}-{args.algo}-E{args.entropy}-{args.seed}"
    else:
        if args.algo == 'codac':
            run_name = f"online-{args.algo}-{args.risk_type}{args.risk_param}-E{args.entropy}-{args.seed}"
        else:
            run_name = f"online-{args.algo}-E{args.entropy}-{args.seed}"

    # Initial environment
    args.entropy_tuning = False
    if args.entropy == "true":
        args.entropy_tuning = True
   
    if args.env == "riskymass":
        from env.risky_pointmass import PointMass
        env = PointMass(risk_prob=args.risk_prob, risk_penalty=args.risk_penalty)
        args.epoch_length = 100
        args.num_epoch = 100
    elif args.env == 'AntObstacle-v0':
        import env
        env = gym.make(args.env)
        env.set_risk(args.risk_prob, args.risk_penalty)
        args.epoch_length = 200
        args.num_epoch = 5000
        args.eval_n_episodes = 100
    else:
        args.num_epoch = 1000
        env, dataset = load_d4rl_dataset(args.env)

    os.makedirs(f'saved_policies/{args.env}/online', exist_ok=True)
    os.makedirs(f'dataset/{args.env}', exist_ok=True)

    # only use batch data for model-free methods
    if args.algo in ['sac', 'cql', 'codac']:
        args.real_ratio = 1.0
    
    if args.wandb:
        wandb.init(project='codac',
                   group=args.env,
                   name=run_name,
                   config=args)
    args.run_name = run_name

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.algo == 'sac':
        agent = SAC(env.observation_space.shape[0], env.action_space,
                    automatic_entropy_tuning=False)
    elif args.algo == 'codac':
        from distributional.codac import CODAC
        agent = CODAC(env.observation_space.shape[0], env.action_space,
                      risk_type=args.risk_type, risk_param=args.risk_param,
                      dist_penalty_type=args.dist_penalty_type)
    elif args.algo == 'cql':
        from sac.cql import CQL
        agent = CQL(env.observation_space.shape[0], env.action_space)


    # Replay Buffer
    env_model = None
    env_pool = ReplayMemory(args.replay_size)
    model_pool = ReplayMemory(1)

    # Imaginary Environment
    predict_env = PredictEnv(env_model, args.env)

    # Sampler Environment
    env_sampler = EnvSampler(env, max_path_length=args.epoch_length)

    # Train
    train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()