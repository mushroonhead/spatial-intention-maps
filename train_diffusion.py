# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import argparse
import random
import sys
from collections import namedtuple
from pathlib import Path

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils

from envs import VectorEnv
# from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler
from diffusers.training_utils import EMAModel
from agents.diffusion2 import DiffusionPolicy, TDErrorQMapDiffTrainer
from intention_encoder_trainers import IntEncoderTrainer
import networks
# from resnet import resnet18
from typing import Iterable, Tuple

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

class TransitionTracker:
    def __init__(self, initial_state):
        self.num_buffers = len(initial_state)
        self.prev_state = initial_state
        self.prev_action = [[None for _ in g] for g in self.prev_state]

    def update_action(self, action):
        for i, g in enumerate(action):
            for j, a in enumerate(g):
                if a is not None:
                    self.prev_action[i][j] = a

    def update_step_completed(self, reward, state, done):
        transitions_per_buffer = [[] for _ in range(self.num_buffers)]
        for i, g in enumerate(state):
            for j, s in enumerate(g):
                if s is not None or done:
                    if self.prev_state[i][j] is not None:
                        transition = (self.prev_state[i][j], self.prev_action[i][j], reward[i][j], s)
                        transitions_per_buffer[i].append(transition)
                    self.prev_state[i][j] = s
        return transitions_per_buffer

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg

#### policy utils ###

def build_diff_trainer(cfg, robot_type,
                       height=96, width=96, num_diffusion_iter=100):
    tensor_kwargs = {'device':device,'dtype':torch.float32}
    # dim
    state_channel = cfg.num_input_channels
    action_channel = VectorEnv.get_num_output_channels(robot_type)
    # net
    conditional_encoder = networks.LightWeightBottleneck(state_channel).to(device)
    noise_model = networks.LightConditionalNetwork(
        VectorEnv.get_num_output_channels(robot_type),
        256).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iter,
        beta_schedule='squaredcos_cap_v2',
        prediction_type='epsilon')
    ema = EMAModel(
        parameters=noise_model.parameters(),
        power=0.75)
    policy = DiffusionPolicy(
        inp_dims=(state_channel, height, width),
        output_dims=(action_channel, height, width),
        num_diffusion_iter=num_diffusion_iter,
        noise_model=noise_model,
        conditional_encoder=conditional_encoder,
        noise_scheduler=noise_scheduler,
        ema=ema,
        tensor_kwargs=tensor_kwargs).to(device)
    # trainer
    training_start = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    num_training_steps = np.ceil((cfg.total_timesteps - training_start)/ cfg.train_freq).astype(np.uint32)
    trainer = TDErrorQMapDiffTrainer(
        diff_model=policy,
        discount_factor=0.9,
        num_training_steps=num_training_steps,
        target_transfer_period=cfg.target_update_freq)

    return policy, trainer

def query_actions(states: Iterable[Iterable],
                  policies: Iterable[torch.nn.Module], robot_types: Iterable[str],
                  exploration_eps: float,
                  debug: bool=False) -> Iterable:
    """
    - Inputs:
        - states: [[states],...]
            - 1st level: robot group
            - 2nd level: robot in robot goup
        - policies: list of policies, in same order as states 1st level
        - robot_types: list of robot group names, in same order as states 1st level
        - exploration_eps: float, chance of a random action
        - debug: bool, true to generate debug info
    - Returns:
        - actions: [[actions],...]
            - 1st level: robot group
            - 2nd level: robot in robot goup
            - action: int, mapping to actual actions availble to that class of robots
        - info: dict, debug info
    """
    actions = []
    outputs = []

    #TODO: change this into a batch evaluation

    with torch.no_grad():
        for group_i_states, policy, robot_type in zip(states, policies, robot_types):
            group_i_actions = []
            group_i_outputs = []
            for state in group_i_states:
                action = None
                output = None
                if state is not None:
                    random_explore = random.random() < exploration_eps
                    if debug or not random_explore:
                        output = policy(torch.tensor(state, device=device).permute(2,0,1).unsqueeze(0)).squeeze(0)
                    action = random.randrange(VectorEnv.get_action_space(robot_type)) \
                        if random_explore \
                        else output.view(-1).argmax().item()
                    if debug or not random_explore:
                        output = output.cpu().numpy()

                group_i_actions.append(action)
                group_i_outputs.append(output)
            actions.append(group_i_actions)
            outputs.append(group_i_outputs)

    info = {'output':outputs} if debug else {}

    return actions, info

#### intention encoder utils ###

def build_int_trainer(cfg):
    # net
    intention_net = networks.FCN((cfg.num_input_channels - 1), 1)
    # trainer
    trainer = IntEncoderTrainer(
        intention_net,
        optim_params={'lr':cfg.learning_rate})

    return intention_net, trainer

def encode_intentions(states: Iterable[Iterable],
                      intention_nets: Iterable[torch.nn.Module],
                      debug=False) -> Tuple[Iterable, dict]:
    """
    Replaces the last layer of state with encoded intention
    - Inputs:
        - states: [[states],...]
            - 1st level: robot group
            - 2nd level: robot in robot goup
        - intention_nets: list of intention_encoders, in same order as states 1st level
        - debug: bool, true to output debug information in Info
    - Returns:
        - states_w_int: [[state_w_int],...]
            - 1st level: robot group
            - 2nd level: robot in robot goup
            - state_w_int: ndarray, intention as last layer of state
        - info: dict, debug info
    """
    states_w_int = []
    out_int = []
    with torch.no_grad():
        for group_i_states, intention_net in zip(states, intention_nets):
            group_i_new_states = []
            group_i_ints = []
            for state in group_i_states:
                if state is None:
                    updated_state = None
                    output_int = None
                else:
                    output_int: torch.Tensor = intention_net(
                        torch.tensor(state[...,:-1], device=device)).sigmoid().squeeze(0).squeeze(0).cpu().numpy()
                    updated_state = np.concatenate(
                        (state[...,:-1], np.expand_dims(output_int, 2)),
                        axis=2)
                group_i_new_states.append(updated_state)
                group_i_ints.append(output_int.cpu().numpy())
            states_w_int.append(group_i_new_states)
            out_int.append(group_i_ints)

    info = {'output_intention': out_int} if debug else {}

    return states_w_int, info

########

def main(cfg, log_scalars=True, log_visuals=True):
    # Set up logging and checkpointing
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    # Create environment
    kwargs = {}
    if cfg.show_gui:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use('agg')
    if cfg.use_predicted_intention:  # Enable ground truth intention map during training only
        kwargs['use_intention_map'] = True
        kwargs['intention_map_encoding'] = 'ramp'
    env = utils.get_env_from_cfg(cfg, **kwargs)

    robot_group_types = env.get_robot_group_types()
    num_robot_groups = len(robot_group_types)

    # build policy trainers for each robot group
    policies, policy_trainers = zip(*[build_diff_trainer(cfg, robot_type) for robot_type in robot_group_types])

    # build intention encoder for each robot group (if using predicted)
    if cfg.use_predicted_intention:
        int_encoders, int_enc_trainers = zip(*[build_int_trainer(cfg) for _ in robot_group_types])

    # Replay buffers for each robot group
    replay_buffers = []
    for _ in range(num_robot_groups):
        replay_buffers.append(ReplayBuffer(cfg.replay_buffer_size))

    # time step and ep (plus any checkpt loading)
    start_timestep = 0
    episode = 0
    training_iter = 0

    # Logging
    meters = Meters()
    if log_scalars:
        train_summary_writer = SummaryWriter(log_dir=str(log_dir / 'train'))
    if log_visuals:
        visualization_summary_writer = SummaryWriter(log_dir=str(log_dir / 'visualization'))

    # Prepare simulation, calculate when to start
    states = env.reset()
    transition_tracker = TransitionTracker(states)
    learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps

    # RL loop, consists of exploration and training
    for timestep in (pbar:= tqdm(range(start_timestep, total_timesteps_with_warm_up),
                         initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout)):

        ################################################################################
        ### Simulation

        # calculate exploration eps
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))

        # if use predicted intention, we override the ground truth intention with predicted
        # - this only fully happens after a few steps of training (ie training with gt first then use predicted)
        if cfg.use_predicted_intention:
            use_ground_truth_intention = max(0, timestep - learning_starts) / cfg.total_timesteps <= cfg.use_predicted_intention_frac
            if not use_ground_truth_intention:
                states, _ = encode_intentions(states, int_encoders)

        # query actions to use
        actions, _ = query_actions(states, policies, robot_group_types, exploration_eps)
        transition_tracker.update_action(actions)

        # Step the simulation
        states, reward, done, info = env.step(actions)

        # Store in buffers
        transitions_per_buffer = transition_tracker.update_step_completed(reward, states, done)
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                replay_buffers[i].push(*transition)

        # Reset if episode ended
        if done:
            states = env.reset()
            transition_tracker = TransitionTracker(states)
            episode += 1

        ################################################################################
        ### Training

        # train net work only after a given time and train
        if timestep >= 100:
        # if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            all_train_info = {}

            for i in range(num_robot_groups):

                # format data from replay buffers
                batch = replay_buffers[i].sample(cfg.batch_size)

                batch_state_map = torch.stack(
                    [torch.tensor(s, device=device).permute(2,0,1) for s in batch.state])           # (B, C, 96, 96)
                batched_actions = torch.tensor(batch.action, dtype=torch.long, device=device)       # (B,)
                batched_rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)    # (B,)
                non_final_next_states = torch.stack(
                    [torch.tensor(s, device=device).permute(2,0,1) for s in batch.next_state \
                     if s is not None])                                                             # (<=B, C, 96, 96)
                non_final_state_mask = torch.tensor([s is not None for s in batch.next_state])      # (B,)

                # train policies
                train_info = policy_trainers[i].train(training_iter,
                                                    batch_state_map, batched_actions,
                                                    batched_rewards, non_final_next_states,
                                                    non_final_state_mask)

                # train intention encoders
                if cfg.use_predicted_intention:
                    train_info_intention = int_enc_trainers[i].train(training_iter,
                                                                     batch_state_map[:,:-1,:,:],
                                                                     batch_state_map[:,-1,:,:].unsqueeze(-3))
                    train_info.update(train_info_intention)

                # accumulate training info
                for name, val in train_info.items():
                    all_train_info['{}/robot_group_{:02}'.format(name, i + 1)] = val

                # update training iteration
                training_iter += 1

        ################################################################################
        # Logging

        # Meters
        if (timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0):
            for name, val in all_train_info.items():
                meters.update(name, val)

            pbar.set_description(f"Avg Total Loss: {meters.avg('loss/robot_group_01'):05f},TD_Loss: {meters.avg('td_loss/robot_group_01'):05f},Diff_Loss: {meters.avg('diff_loss/robot_group_01'):05f}")

        if done:

            # log scenario params
            if log_scalars:
                for name in meters.get_names():
                    train_summary_writer.add_scalar(name, meters.avg(name), timestep + 1)
                meters.reset()

                train_summary_writer.add_scalar('steps', info['steps'], timestep + 1)
                train_summary_writer.add_scalar('total_cubes', info['total_cubes'], timestep + 1)
                train_summary_writer.add_scalar('episodes', episode, timestep + 1)

                for i in range(num_robot_groups):
                    for name in ['cumulative_cubes', 'cumulative_distance', 'cumulative_reward', 'cumulative_robot_collisions']:
                        train_summary_writer.add_scalar('{}/robot_group_{:02}'.format(name, i + 1), np.mean(info[name][i]), timestep + 1)

            # visualize network outputs
            if log_visuals and (timestep >= learning_starts):
                # choose a random state from the buffer
                random_states = [[random.choice(replay_buffers[i].buffer).state] for _ in range(num_robot_groups)]
                # run an encoding step
                if cfg.use_predicted_intention:
                    random_states, enc_info = encode_intentions(random_states, int_encoders, debug=True)
                _, act_info = query_actions(random_states, policies, robot_group_types, exploration_eps=0.0, debug=True)
                for i in range(num_robot_groups):
                    visualization = utils.get_state_output_visualization(
                        random_states[i][0], act_info['output'][i][0]).transpose((2, 0, 1))
                    visualization_summary_writer.add_image('output/robot_group_{:02}'.format(i + 1), visualization, timestep + 1)
                    if cfg.use_predicted_intention:
                        visualization_intention = utils.get_state_output_visualization(
                            random_states[i][0],
                            np.stack((random_states[i][0][:, :, -1], enc_info['output_intention'][i][0]), axis=0)  # Ground truth and output
                        ).transpose((2, 0, 1))
                        visualization_summary_writer.add_image('output_intention/robot_group_{:02}'.format(i + 1), visualization_intention, timestep + 1)

        ################################################################################
        # Checkpointing

        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # save state dicts
            all_state_dicts_filename = 'all_state_dicts_{:08d}.pth.tar'.format(timestep + 1)
            checkpoint_path = checkpoint_dir / all_state_dicts_filename
            checkpoint = {
                'timestep': timestep + 1,
                'episode': episode,
                'policy_trainers': [policy_trainer.get_state_dicts() for policy_trainer in  policy_trainers],
                'replay_buffers': [replay_buffers[i] for i in range(num_robot_groups)],
            }
            if cfg.use_predicted_intention:
                checkpoint['int_net_trainers'] = [int_net_trainer.get_state_dicts() for int_net_trainer in int_enc_trainers]
            torch.save(checkpoint, str(checkpoint_path))

            # Save updated config file
            cfg.checkpoint_path = str(checkpoint_path)
            utils.save_config(log_dir / 'config.yml', cfg)

            # Remove old checkpoint
            checkpoint_paths = list(checkpoint_dir.glob('all_state_dicts_*.pth.tar'))
            [old_checkpoint_path.unlink() for old_checkpoint_path in checkpoint_paths if
                not (old_checkpoint_path == checkpoint_path)]

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    config_path = parser.parse_args().config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is not None:
        config_path = utils.setup_run(config_path)
        main(utils.load_config(config_path))