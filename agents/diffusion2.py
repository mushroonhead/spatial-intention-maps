"""
An alternative diffusion model using huggingface diffuser lib
"""
import torch
from copy import deepcopy
from math import prod
from diffusers.training_utils import EMAModel
from diffusers.schedulers import SchedulerMixin
from diffusers.optimization import get_scheduler
from typing import Optional

class DiffusionPolicy(torch.nn.Module):
    def __init__(self,
                 inp_dims: torch.Size, output_dims: torch.Size,
                 num_diffusion_iter: int,
                 noise_model: torch.nn.Module,
                 conditional_encoder: torch.nn.Module,
                 noise_scheduler: SchedulerMixin,
                 ema: Optional[EMAModel],
                 tensor_kwargs: dict,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inp_dims = inp_dims
        self.output_dims = output_dims
        self.num_diffusion_iter = num_diffusion_iter
        self.noise_model = noise_model
        self.conditional_encoder = conditional_encoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iter)
        self.ema = ema
        self.tensor_kwargs = tensor_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x: (...,inp_dim..)  tensor
        - Returns:
            - out: (...,out_dim...) tensor
        """
        # get sizes
        batch_size = x.shape[:-len(self.inp_dims)]

        # initialize image from Gaussian noise
        q_map = torch.randn(
            (*batch_size, *self.output_dims),
            device=x.device, dtype=x.dtype)

        # conditional encoding
        cond = self.conditional_encoder(x) # (B, seq, feature_dim)

        # denoising
        for k in self.noise_scheduler.timesteps:

            # predict noise
            noise_pred = self.noise_model(
                sample=q_map,
                timestep=k,
                encoder_hidden_states=cond)

            # inverse diffusion step (remove noise)
            q_map = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=q_map).prev_sample

        return q_map

    def step_ema(self):
        if self.ema is not None:
            self.ema.step(self.noise_model.parameters())

    def get_state_dicts(self) -> dict:
        state_dict = {'noise_model':self.noise_model.state_dict(),
                      'cond_encoder':self.conditional_encoder.state_dict()}
        if self.ema is not None:
            state_dict['ema'] = self.ema.state_dict()

        return state_dict

class QMapDiffTrainerBase:
    """
    Base Qmap trainer for Diffusion Policy
    """
    def __init__(self,
                 diff_model: DiffusionPolicy,
                 discount_factor: float,
                 num_training_steps: int,
                 diff_loss: torch.nn.Module = torch.nn.MSELoss(),
                 optim_type=torch.optim.AdamW,
                 optim_params={'lr':1e-4, 'weight_decay':1e-6}
                 ) -> None:
        # policy
        self.policy = diff_model
        # training obj
        self.optimizer = optim_type(
            params=self.policy.parameters(),
            **optim_params)
        # lr scheduler
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps)
        # q value discount factor
        self.discount_factor = discount_factor
        # losses
        self.diff_loss = diff_loss

    def get_diffusion_loss(self, output_target: torch.Tensor, inp_cond: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - output_target: (...,output_dims...) tensor
            - inp_cond: (...,inp_dims...) tensor
        - Returns:
            - loss: (...,) tensor
        """

        # get batch_shape and resize
        output_target = output_target.view(-1, *self.policy.output_dims)
        inp_cond = inp_cond.view(-1, *self.policy.inp_dims)

        # sample noise
        noise = torch.randn(output_target.shape, device=output_target.device, dtype=output_target.dtype)

        # random timesteps
        timesteps = torch.randint(
            0, self.policy.noise_scheduler.num_train_timesteps,
            (output_target.shape[0],), device=output_target.device, dtype=torch.int32)

        # add noise dep on timestep to output_target
        noisy_output = self.policy.noise_scheduler.add_noise(
            output_target, noise, timesteps)

        # predict the noise residual
        noise_pred = self.policy.noise_model(
            sample=noisy_output, timestep=timesteps,
            encoder_hidden_states=self.policy.conditional_encoder(inp_cond))

        # return loss
        return self.diff_loss(noise_pred, noise)

    def get_state_dicts(self) -> dict:
        return {'policy':self.policy.get_state_dicts(),
                'policy_optim':self.optimizer.state_dict(),
                'policy_lr':self.lr_scheduler.state_dict()}


class TDErrorQMapDiffTrainer(QMapDiffTrainerBase):
    """
    Treats output of policy as a qvalue map
    """
    def __init__(self,
                 diff_model: DiffusionPolicy,
                 discount_factor: float,
                 num_training_steps: int,
                 target_transfer_period: int,
                 qmap_loss: torch.nn.Module= torch.nn.SmoothL1Loss(),
                 diff_loss: torch.nn.Module = torch.nn.MSELoss()) -> None:
        super().__init__(diff_model, discount_factor, num_training_steps, diff_loss)
        self.action_channels, self.height, self.width = diff_model.output_dims
        self.target_policy = deepcopy(diff_model)
        self.target_policy.eval()
        self.target_transfer_period = target_transfer_period
        self.qmap_loss = qmap_loss

    def train(self, training_iter: int,
              state_map: torch.Tensor, actions: torch.Tensor,
              rewards: torch.Tensor, non_final_next_states: torch.Tensor,
              non_final_state_mask: torch.BoolTensor) -> dict:
        """
        - Inputs:
            - state_map: (B,C,H,W) tensor
            - actions: (B,) tensor
            - rewards: (B,) tensor
            - non_final_next_states: (<=B,C,H,W) tensor
            - non_final_state_mask: (B,) bool tensor
            - logger: optional SumaryWriter
        - Returns:
            - info: dict, any loss or metric info to be tracked
        """
        B = state_map.shape[0]
        device = state_map.device

        # standard dqn td error training (here we assume r to be determined only on s,a)
        with torch.no_grad():
            next_step_vals = self.target_policy(non_final_next_states).detach()
            next_state_shape = next_step_vals.shape[1:]
            target_q_map = rewards[:,None,None,None] + (self.discount_factor * \
                torch.eye(B, device=device)[non_final_state_mask].T @ next_step_vals.view(-1,prod(next_state_shape))
                ).view(B,*next_state_shape)
        pred_q_map = self.policy(state_map)

        td_loss: torch.Tensor = self.qmap_loss(pred_q_map, target_q_map)

        # diffusion loss # doesnt make sense
        diff_loss = self.get_diffusion_loss(pred_q_map.detach(), state_map)

        # backpropagation
        loss = td_loss + diff_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.policy.step_ema()

        # transfer weights at interval
        if training_iter % self.target_transfer_period == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        # losses
        return {'td_loss':td_loss.item(),
                'diff_loss':diff_loss.item(),
                'loss':loss.item()}

    def get_state_dicts(self) -> dict:
        state_dict = super().get_state_dicts()
        state_dict['target_policy'] = self.target_policy.get_state_dicts()

        return state_dict


class TwinQNetwork(torch.nn.Module):
    def __init__(self,
                 critic_1: torch.nn.Module,
                 critic_2: torch.nn.Module,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.critic_1 = critic_1
        self.critic_2 = critic_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_1(x), self.critic_2(x)


class DQCriticQMapDiffTrainer(QMapDiffTrainerBase):
    """
    Treats output of policy as a qvalue map
    """
    def __init__(self,
                 diff_model: DiffusionPolicy,
                 critic_model_1: torch.nn.Module,
                 critic_model_2: torch.nn.Module,
                 discount_factor: float,
                 num_training_steps: int,
                 critic_xfer_period: int,
                 critic_tau: float,
                 qmap_loss: torch.nn.Module= torch.nn.SmoothL1Loss(),
                 diff_loss: torch.nn.Module = torch.nn.MSELoss()) -> None:
        super().__init__(diff_model, discount_factor, num_training_steps, diff_loss)
        self.height = diff_model.height
        self.width = diff_model.width
        self.action_channels = diff_model.out_channel
        self.critic = TwinQNetwork(critic_model_1, critic_model_2)
        self.critic_target = deepcopy(self.critic).eval()
        self.target_transfer_period = critic_xfer_period
        self.critic_tau = critic_tau
        self.qmap_loss = qmap_loss
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=1e-4)

    def train(self, training_iter: int,
              state_map: torch.Tensor, actions: torch.Tensor,
              rewards: torch.Tensor, non_final_next_states: torch.Tensor,
              non_final_state_mask: torch.BoolTensor) -> dict:
        """
        - Inputs:
            - state_map: (B,C,H,W) tensor
            - actions: (B,) tensor
            - rewards: (B,) tensor
            - non_final_next_states: (<=B,C,H,W) tensor
            - non_final_state_mask: (B,) bool tensor
            - logger: optional SumaryWriter
        - Returns:
            - info: dict, any loss or metric info to be tracked
        """

        ## q learning

        current_q1, current_q2 = self.critic(state_map)

        def calculate_target_q(reward: torch.Tensor, next_nonfinal_state_q: torch.Tensor,
                               non_final_state_mask: torch.BoolTensor, discount_factor: float):
            """
            - reward: (B,)
            - next_nonfinal_state_q: (>=B,...)
            - non_final_state_mask: (B,)
            """
            # reshape first
            B = next_nonfinal_state_q.shape[0]
            orginal_shape = next_nonfinal_state_q.shape[1:]
            next_nonfinal_state_q = next_nonfinal_state_q.view(B,-1)
            device = reward.device

            # add next_nonfinal_state_q
            reward = reward.view(B, *[1 for _ in orginal_shape]) # (B,1...)
            target_q = reward + discount_factor * \
                (torch.eye(B, device=device)[non_final_state_mask].T @ next_nonfinal_state_q).view(B,*orginal_shape)

            return target_q

        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(non_final_next_states).detach()
            target_q1 = calculate_target_q(rewards, next_q1, non_final_state_mask,
                                           self.discount_factor)
            target_q2 = calculate_target_q(rewards, next_q2, non_final_state_mask,
                                           self.discount_factor)
            target_q = torch.minimum(target_q1, target_q2)

        critic_td_error: torch.Tensor = self.qmap_loss(current_q1, target_q) + self.qmap_loss(current_q2, target_q)

        # possible gradient clipping

        critic_td_error.backward()
        self.critic_optim.step()
        self.critic_optim.zero_grad()

        if (training_iter +1) % self.critic_xfer_period == 0:
            # soft policy update for critic
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)


        ## diffusion learning

        # diffusion loss # doesnt make sense but oh well..
        bc_loss = self.get_diffusion_loss(state_map, torch.minimum(current_q1, current_q2).detach())

        # backpropagation
        bc_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.policy.step_ema()

        # losses
        return {'critic_td_error':critic_td_error.item(),
                'bc_loss':bc_loss.item()}
