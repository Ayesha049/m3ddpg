import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import requests
import json

API_KEY = ""

def gpt_call(prompt):
    # url = "https://api.openai.com/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer {}".format(API_KEY)
    # }
    # data = {
    #     "model": "gpt-3.5-turbo",  # updated supported model
    #     "messages": [
    #         {"role": "system", "content": "You are an adversarial perturbation generator for robust RL. Output only the revised observation as a Python list."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     "temperature": 0.7,
    #     "max_tokens": 200
    # }

    # try:
    #     response = requests.post(url, headers=headers, data=json.dumps(data))
    #     result = response.json()

    #     if "error" in result:
    #         print("OpenAI API error:", result["error"])
    #         return None

    #     if "choices" not in result or len(result["choices"]) == 0:
    #         print("OpenAI API returned no choices:", result)
    #         return None

    #     # gpt-3.5-turbo returns content here:
    #     return result["choices"][0]["message"]["content"].strip()

    # except Exception as e:
    #     print("GPT call failed:", str(e))
    #     return None

    return None



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--bad-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="", help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    parser.add_argument("--load-dir", type=str, default="./model", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    # --- Robustness settings ---
    parser.add_argument("--noise-factor", type=str, default="state", choices=["none", "state", "reward"],
                        help="where to apply noise (state/reward/none)")
    parser.add_argument("--noise-type", type=str, default="gauss", choices=["gauss", "shift", "uniform"],
                        help="type of noise distribution")
    parser.add_argument("--noise-mu", type=float, default=0.0, help="mean for Gaussian noise")
    parser.add_argument("--noise-sigma", type=float, default=0.1, help="std for Gaussian noise")
    parser.add_argument("--noise-shift", type=float, default=0.05, help="shift noise magnitude")
    parser.add_argument("--uniform-low", type=float, default=-0.1, help="low bound for uniform noise")
    parser.add_argument("--uniform-high", type=float, default=0.1, help="high bound for uniform noise")
    parser.add_argument("--llm-disturb-interval", type=int, default=5, help="steps between disturbances")

    # --- LLM-guided adversary ---
    parser.add_argument("--llm-guide", type=str, default="adversary", choices=["none", "adversary"],
                        help="enable LLM-guided perturbations")
    parser.add_argument("--llm-guide-type", type=str, default="stochastic",
                        choices=["stochastic", "uniform", "constraint"],
                        help="LLM adversarial perturbation type")


    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        print("{} bad agents".format(i))
        policy_name = arglist.bad_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        print("{} good agents".format(i))
        policy_name = arglist.good_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    return trainers


def train(arglist):
    if arglist.test:
        np.random.seed(71)
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and bad policy {} with {} adversaries'.format(arglist.good_policy, arglist.bad_policy, num_adversaries))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
            if arglist.load_name == "":
                # load seperately
                bad_var_list = []
                for i in range(num_adversaries):
                    bad_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(bad_var_list)
                U.load_state(arglist.load_bad, saver)

                good_var_list = []
                for i in range(num_adversaries, env.n):
                    good_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(good_var_list)
                U.load_state(arglist.load_good, saver)
            else:
                print('Loading previous state from {}'.format(arglist.load_name))
                U.load_state(arglist.load_name)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if not arglist.test:
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, global_step = len(episode_rewards), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("{} vs {} steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(arglist.bad_policy, arglist.good_policy,
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                suffix = '_test.pkl' if arglist.test else '.pkl'
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards' + suffix
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards' + suffix

                if not os.path.exists(os.path.dirname(rew_file_name)):
                    try:
                        os.makedirs(os.path.dirname(rew_file_name))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise

                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break




def test(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.bad_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        #if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir)

        # Parameters for testing
        n_episodes = 10
        max_episode_len = arglist.max_episode_len

        all_rewards = []
        print('Starting testing...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)
            for step in range(max_episode_len):
                # get actions from trained policies
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, _ = env.step(action_n)

                episode_reward += rew_n
                obs_n = new_obs_n

                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))



def testRobustness(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.bad_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        #if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir)

        # Testing params
        n_episodes = 10
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):
                # get actions
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)

                # === Apply your disruption here ===
                disrupted_obs_n = []
                for i, obs in enumerate(new_obs_n):
                    disrupted_obs_n.append(apply_observation_disruption(
                        obs, rew_n[i], env, arglist
                    ))

                # track reward
                episode_reward += rew_n
                obs_n = disrupted_obs_n

                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))


def testRobustnessOA(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.bad_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir)

        # Testing params
        n_episodes = 10
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):
                # --- Apply observation disruption before action selection ---
                obs_n_disrupted = [
                    apply_observation_disruption(obs, 0, env, arglist)
                    for obs in obs_n
                ]

                # --- Get actions from agents ---
                action_n = [
                    agent.action(obs_dis)
                    for agent, obs_dis in zip(trainers, obs_n_disrupted)
                ]

                # --- Apply action disruption ---
                action_n_disrupted = [
                    apply_action_disruption(action, 0, env, arglist)
                    for action in action_n
                ]

                # --- Environment step ---
                new_obs_n, rew_n, done_n, info_n = env.step(action_n_disrupted)

                # --- Track reward ---
                episode_reward += rew_n
                obs_n = new_obs_n

                # --- Render if needed ---
                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))


def apply_observation_disruption(observation, reward, env, args):
    """
    Apply noise or LLM-guided disruption to observation/reward.
    Ensures the output is always a valid NumPy array.
    """
    # Keep track of iterations
    env.llm_disturb_iteration += 1
    obs_orig = np.array(observation, dtype=np.float32)

    # === Apply noise ===
    if args.noise_factor == "state" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        if args.noise_type == "gauss":
            noise = np.random.normal(args.noise_mu, args.noise_sigma, size=obs_orig.shape)
            obs_orig = obs_orig + noise
        elif args.noise_type == "shift":
            obs_orig = obs_orig + args.noise_shift
        elif args.noise_type == "uniform":
            noise = np.random.uniform(args.uniform_low, args.uniform_high, size=obs_orig.shape)
            obs_orig = obs_orig + noise

    # === LLM-guided perturbation ===
    if args.llm_guide == "adversary" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        prompt = (
            "Robust RL adversary: Current reward = {}, previous reward = {}. "
            "Revise this state: {}. Output only the revised state as a Python list."
        ).format(reward, env.previous_reward, obs_orig.tolist())

        obs_from_gpt = gpt_call(prompt)
        if obs_from_gpt is not None:
            try:
                obs_from_gpt = np.array(eval(obs_from_gpt), dtype=np.float32)
                # reshape/pad/truncate to match original observation
                if obs_from_gpt.shape != obs_orig.shape:
                    flat = obs_from_gpt.flatten()
                    size_needed = np.prod(obs_orig.shape)
                    if flat.size < size_needed:
                        flat = np.pad(flat, (0, size_needed - flat.size), mode="constant")
                    else:
                        flat = flat[:size_needed]
                    obs_from_gpt = flat.reshape(obs_orig.shape)
                obs_orig = obs_from_gpt
            except:
                # fallback to original observation
                pass

    env.previous_reward = reward
    return obs_orig


def apply_action_disruption(action, reward, env, args):
    env.llm_disturb_iteration += 1
    action_orig = np.array(action, dtype=np.float32)

    if args.noise_factor == "action" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        if args.noise_type == "gauss":
            action_orig = action_orig + np.random.normal(args.noise_mu, args.noise_sigma, size=action_orig.shape)
        elif args.noise_type == "shift":
            action_orig = action_orig + args.noise_shift
        elif args.noise_type == "uniform":
            action_orig = action_orig + np.random.uniform(args.uniform_low, args.uniform_high, size=action_orig.shape)

    if args.llm_guide == "adversary" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        prompt = (
            "Robust RL adversary: Current reward = {}, previous reward = {}. "
            "Revise this action: {}. Output only the revised action as a Python list."
        ).format(reward, env.previous_reward, action_orig.tolist())
        action_from_gpt = gpt_call(prompt)
        if action_from_gpt is not None:
            try:
                action_from_gpt = np.array(eval(action_from_gpt), dtype=np.float32)
                if action_from_gpt.shape != action_orig.shape:
                    flat = action_from_gpt.flatten()
                    size_needed = np.prod(action_orig.shape)
                    if flat.size < size_needed:
                        flat = np.pad(flat, (0, size_needed - flat.size), mode="constant")
                    else:
                        flat = flat[:size_needed]
                    action_from_gpt = flat.reshape(action_orig.shape)
                action_orig = action_from_gpt
            except:
                pass

    env.previous_reward = reward
    return action_orig

if __name__ == '__main__':
    arglist = parse_args()
    # train(arglist)
    testRobustnessOA(arglist)
