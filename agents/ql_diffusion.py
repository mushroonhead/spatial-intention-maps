# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
#from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

import resnet


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fcn_input_channels, fcn_output_channels, one_channel_state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=fcn_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, fcn_output_channels, kernel_size=1, stride=1)

        self.q1_model = nn.Sequential(nn.Linear(fcn_output_channels*one_channel_state_dim  + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(fcn_output_channels*one_channel_state_dim  + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
    
    def state_through_resnet(self, state):
        state = self.resnet18.features(state)
        state = self.conv1(state)
        state = self.bn1(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv3(state)
        if state.shape[1] != 1 or state.shape[2] != 96 or state.shape[3] != 96:
            raise Exception("ERROR: state dimension is not batch_size * 1 * 96 * 96, error is in model.py")

        return state

    def forward(self, state, action):
        #print(state.shape,action.shape)
        x = self.state_through_resnet(state)
        x = torch.cat([x.flatten(start_dim=-3), action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        #x = torch.cat([state.flatten(start_dim=-3), action], dim=-1)
        x = self.state_through_resnet(state)
        x = torch.cat([x.flatten(start_dim=-3), action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 fcn_input_channels,
                 fcn_output_channels,
                 one_channel_state_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, 
                        fcn_input_channels=fcn_input_channels, fcn_output_channels=fcn_output_channels,
                        one_channel_state_dim=one_channel_state_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim,fcn_input_channels,fcn_output_channels,one_channel_state_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.1)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.1)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

        self.transform = transforms.ToTensor()

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    
    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        #metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            #state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            batch = replay_buffer.sample(batch_size)
            #print(batch.action)
            state = torch.cat([self.apply_transform(s) for s in batch.state]).to(self.device)  # (32, 4, 96, 96)
            action_indx = torch.tensor(batch.action, dtype=torch.long).to(self.device)  # (32,)
            reward = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)  # (32,)

            next_state_not_done = torch.cat([self.apply_transform(s) for s in batch.next_state if s is not None]).to(self.device)  #(<=32)
            not_done = [s is not None for s in batch.next_state]
        
                #torch.cat([1.0 if s is not None else 0.0]).to(self.device)

            action = torch.zeros((action_indx.shape[0], 96*96), dtype=torch.float32,device = self.device)
            for i in range(action_indx.shape[0]):
                action[i,action_indx[i]] = 1.0
            

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state_not_done, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1_not_done, target_q2_not_done = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1_not_done = target_q1_not_done.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2_not_done = target_q2_not_done.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q_not_done = torch.min(target_q1_not_done, target_q2_not_done)
            else:
                next_action_not_done = self.ema_model(next_state_not_done)
                target_q1_not_done, target_q2_not_done = self.critic_target(next_state_not_done, next_action_not_done)
                #print(next_action.shape,next_state.shape, target_q1.shape,target_q2.shape)
                target_q_not_done = torch.min(target_q1_not_done, target_q2_not_done)
                
            #print(reward.shape,target_q.shape,torch.tensor(not_done,device=self.device).shape)
            target_q = reward.unsqueeze(-1)
            #print('t',target_q.shape, target_q_not_done.shape)
            target_q[not_done] += self.discount * target_q_not_done.detach()

            #print(current_q1.shape, target_q.shape)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            """bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()"""

            """ Train policy network only every few iterations """
            if self.step % self.update_ema_every == 0:
                """ Train policy network """ 
                bc_loss = self.actor.loss(action, state)
                new_action = self.actor(state)

                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
                else:
                    q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
                actor_loss = bc_loss + self.eta * q_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_norm > 0: 
                    actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                self.actor_optimizer.step()

                """ Step Target network """  
                self.step_ema()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q_not_done.mean().item(), self.step)

            #metric['actor_loss'].append(actor_loss.item())
            #metric['bc_loss'].append(bc_loss.item())
            #metric['ql_loss'].append(q_loss.item())
            #metric['critic_loss'].append(critic_loss.item())
            metric = {}
            metric['actor_loss'] = actor_loss.item()
            metric['bc_loss'] = bc_loss.item()
            metric['ql_loss'] = q_loss.item()
            metric['critic_loss'] = critic_loss.item()

            print(metric)

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric
    
    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

    """def step(self, state, VectorEnv_action_space, exploration_eps=None):
        action = [[None for _ in g] for g in state]
        with torch.no_grad():
            for i, g in enumerate(state):
                #robot_type = robot_group_types[i]
                #diffusion_models[i].eval()
                self.actor.eval()
                for j, s in enumerate(g):
                    if s is not None:
                        s = transforms.ToTensor()(s).unsqueeze(0).to(device=self.device,dtype=torch.float32)
                        #print(s)
                        o = self.actor.sample(s).squeeze(0)
                        if random.random() < exploration_eps:
                            #a = random.randrange(VectorEnv.get_action_space(robot_type))
                            a = random.randrange(VectorEnv_action_space)
                        else:
                            a = o.view(1, -1).max(1)[1].item()
                        action[i][j] = a
                        #output[i][j] = o.cpu().numpy()
                if self.train:
                    #policy_diffusion[i].train()
                    self.actor.train()
        return action """




