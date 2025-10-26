"""Wavelet shrinkage convolutional layer with D3QN-based kernel selection.

This module implements a functional convolutional layer that maintains a
collection of analytic wavelet kernels and leverages a Dueling Double DQN
agent to adaptively select the most appropriate kernel for each input batch.
After the convolution a soft-thresholding shrinkage operation is applied to
denoise the feature maps and preserve task-relevant components.

The implementation focuses on clarity and modularity so that the layer can be
seamlessly plugged into the existing TFN architecture.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import random
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _softplus_positive(param: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Ensure the tensor is strictly positive via a softplus transform."""

    return F.softplus(param, beta=beta, threshold=threshold) + 1e-6


class DuelingQNetwork(nn.Module):
    """Simple dueling network used inside the D3QN agent."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class D3QNAgent(nn.Module):
    """A lightweight D3QN agent for adaptive kernel selection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.98,
        buffer_size: int = 2048,
        batch_size: int = 64,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        tau: float = 0.01,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.online_net = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.register_buffer("epsilon", torch.tensor(epsilon_start, dtype=torch.float32))
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory: deque = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()

        self.last_q_values: Optional[Tensor] = None

    @property
    def device(self) -> torch.device:
        return next(self.online_net.parameters()).device

    def select_action(self, states: Tensor, training: bool = True) -> Tensor:
        """Select an action for every sample in the batch."""

        states = states.to(self.device)
        batch_size = states.size(0)

        if training and torch.rand(1, device=self.device).item() < float(self.epsilon):
            actions = torch.randint(0, self.action_dim, (batch_size,), device=self.device)
            self.last_q_values = None
        else:
            q_values = self.online_net(states)
            actions = torch.argmax(q_values, dim=1)
            self.last_q_values = q_values.detach()

        if training:
            new_eps = max(self.epsilon_end, float(self.epsilon) * self.epsilon_decay)
            self.epsilon.fill_(new_eps)

        return actions

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """Store a batch of transitions into the replay buffer."""

        state = state.detach().to(self.device)
        next_state = next_state.detach().to(self.device)
        reward = reward.detach().to(self.device)
        action = action.detach().to(self.device)
        done = done.detach().to(self.device)

        for idx in range(state.size(0)):
            self.memory.append(
                (
                    state[idx],
                    action[idx],
                    reward[idx],
                    next_state[idx],
                    done[idx],
                )
            )

    def update(self) -> None:
        """Run one D3QN optimisation step when enough samples are ready."""

        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        state_batch = torch.stack([item[0] for item in transitions]).to(self.device)
        action_batch = torch.stack([item[1] for item in transitions]).long().to(self.device)
        reward_batch = torch.stack([item[2] for item in transitions]).to(self.device)
        next_state_batch = torch.stack([item[3] for item in transitions]).to(self.device)
        done_batch = torch.stack([item[4] for item in transitions]).to(self.device)

        q_values = self.online_net(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            target_values = reward_batch + self.gamma * (1.0 - done_batch) * next_q

        loss = self.loss_fn(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)


class WaveletShrinkageConv1d(nn.Module):
    """Wavelet convolution layer with RL-driven kernel selection and shrinkage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        clamp_flag: bool = True,  # reserved for compatibility
        wavelet_types: Sequence[str] = ("morlet", "mexhat", "laplace"),
        agent_hidden: int = 128,
        agent_gamma: float = 0.98,
        agent_buffer_size: int = 2048,
        agent_batch_size: int = 64,
        agent_lr: float = 1e-3,
        agent_epsilon_start: float = 1.0,
        agent_epsilon_end: float = 0.05,
        agent_epsilon_decay: float = 0.995,
        agent_tau: float = 0.01,
        threshold_init: float = 0.1,
        use_rl: bool = True,
        use_denoising: bool = True,
        fixed_wavelet: str = "morlet",
    ) -> None:
        super().__init__()

        if bias:
            raise ValueError("WaveletShrinkageConv1d does not support bias terms.")

        if kernel_size % 2 == 0:
            raise ValueError("WaveletShrinkageConv1d expects an odd kernel size for symmetry.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wavelet_types = tuple(w.lower() for w in wavelet_types)
        self.num_wavelets = len(self.wavelet_types)
        self.use_rl = bool(use_rl)
        self.use_denoising = bool(use_denoising)
        self.fixed_wavelet = fixed_wavelet.lower() if fixed_wavelet is not None else None

        # Wavelet scale parameters grouped into a single learnable tensor for compatibility
        self.superparams = nn.Parameter(torch.ones(out_channels, in_channels, self.num_wavelets))

        # Learnable thresholds for soft shrinkage
        self.threshold = nn.Parameter(torch.full((out_channels,), float(threshold_init)))

        # Pre-computed symmetric sampling grid (registered as buffer for device management)
        time_grid = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.get_default_dtype())
        self.register_buffer("time_grid", time_grid)

        state_dim = in_channels * 2  # mean & std per channel
        if self.use_rl:
            self.agent = D3QNAgent(
                state_dim=state_dim,
                action_dim=self.num_wavelets,
                hidden_dim=agent_hidden,
                gamma=agent_gamma,
                buffer_size=agent_buffer_size,
                batch_size=agent_batch_size,
                lr=agent_lr,
                epsilon_start=agent_epsilon_start,
                epsilon_end=agent_epsilon_end,
                epsilon_decay=agent_epsilon_decay,
                tau=agent_tau,
            )
            self.register_buffer("fixed_action", torch.tensor(0, dtype=torch.long))
        else:
            self.agent = None
            if self.fixed_wavelet and self.fixed_wavelet in self.wavelet_types:
                fixed_idx = self.wavelet_types.index(self.fixed_wavelet)
            else:
                fixed_idx = 0
                if self.fixed_wavelet and self.fixed_wavelet not in self.wavelet_types:
                    warnings.warn(
                        f"fixed_wavelet '{self.fixed_wavelet}' not in available types {self.wavelet_types}; "
                        f"defaulting to '{self.wavelet_types[fixed_idx]}'",
                        RuntimeWarning,
                    )
            self.register_buffer("fixed_action", torch.tensor(fixed_idx, dtype=torch.long))

        self.agent_cache: Optional[Dict[str, Tensor]] = None
        self.last_actions: Optional[Tensor] = None

    def extra_repr(self) -> str:
        return (
            f"wavelets={self.wavelet_types}, kernel_size={self.kernel_size}, "
            f"out_channels={self.out_channels}, stride={self.stride}, padding={self.padding}, "
            f"use_rl={self.use_rl}, use_denoising={self.use_denoising}"
        )

    def _wavelet_kernel(self, wavelet_idx: int) -> Tensor:
        scale = _softplus_positive(self.superparams[:, :, wavelet_idx])
        scale = scale.unsqueeze(-1)
        grid = self.time_grid.view(1, 1, -1).to(scale.device)
        xi = grid / scale

        if self.wavelet_types[wavelet_idx] == "morlet":
            kernel = torch.exp(-0.5 * xi.pow(2)) * torch.cos(5.0 * xi)
        elif self.wavelet_types[wavelet_idx] == "mexhat":
            pi_tensor = torch.tensor(torch.pi, device=xi.device)
            norm_const = 2.0 / (torch.sqrt(torch.tensor(3.0, device=xi.device)) * pi_tensor.pow(0.25))
            kernel = norm_const * (1.0 - xi.pow(2)) * torch.exp(-0.5 * xi.pow(2))
        elif self.wavelet_types[wavelet_idx] == "laplace":
            kernel = torch.sign(grid) * torch.exp(-torch.abs(xi))
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_types[wavelet_idx]}")

        kernel = kernel - kernel.mean(dim=-1, keepdim=True)
        kernel = kernel / (kernel.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        return kernel

    def _prepare_kernels(self) -> Tensor:
        weights: List[Tensor] = []
        for idx in range(self.num_wavelets):
            weights.append(self._wavelet_kernel(idx))
        weight_stack = torch.stack(weights, dim=0)  # (num_wavelets, out_channels, in_channels, kernel)
        return weight_stack.view(self.num_wavelets * self.out_channels, self.in_channels, self.kernel_size)

    @property
    def weight(self) -> Tensor:
        """Expose dynamic kernels for compatibility with legacy utilities."""

        return self._prepare_kernels()

    def _build_state(self, input_tensor: Tensor) -> Tensor:
        mean_feat = input_tensor.mean(dim=-1)
        std_feat = input_tensor.std(dim=-1, unbiased=False)
        return torch.cat([mean_feat, std_feat], dim=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        kernels = self._prepare_kernels()
        conv_all = F.conv1d(
            input_tensor,
            kernels,
            stride=self.stride,
            padding=self.padding,
            groups=1,
        )
        batch_size, _, seq_len = conv_all.shape
        conv_all = conv_all.view(batch_size, self.num_wavelets, self.out_channels, seq_len)

        if self.use_rl:
            states = self._build_state(input_tensor)
            actions = self.agent.select_action(states, training=self.training)
        else:
            actions = self.fixed_action.expand(batch_size).to(input_tensor.device)
            states = None
        one_hot = F.one_hot(actions, num_classes=self.num_wavelets).view(batch_size, self.num_wavelets, 1, 1)
        selected = (conv_all * one_hot).sum(dim=1)

        if self.use_denoising:
            thresh = _softplus_positive(self.threshold).view(1, -1, 1)
            output = torch.sign(selected) * F.relu(selected.abs() - thresh)
        else:
            output = selected

        if self.training and self.use_rl:
            pooled = F.adaptive_avg_pool1d(selected.detach(), 1).view(batch_size, -1)
            self.agent_cache = {
                "states": states.detach(),
                "actions": actions.detach(),
                "next_states": pooled.detach(),
            }
        else:
            self.agent_cache = None

        self.last_actions = actions.detach()
        return output

    def update_agent(self, reward: float, done: bool = False, loss: Optional[float] = None) -> None:
        if not self.use_rl or self.agent is None:
            return

        if not self.training or self.agent_cache is None:
            return

        states = self.agent_cache["states"]
        actions = self.agent_cache["actions"]
        next_states = self.agent_cache["next_states"]

        reward_value = torch.tensor(float(reward), dtype=torch.float32, device=states.device)
        reward_batch = reward_value.repeat(states.size(0))
        if loss is not None:
            reward_batch = reward_batch - float(loss) * 0.1

        done_tensor = torch.full((states.size(0),), float(done), dtype=torch.float32, device=states.device)

        self.agent.store_transition(states, actions, reward_batch, next_states, done_tensor)
        self.agent.update()
        self.agent_cache = None

    def get_kernel_dictionary(self) -> Dict[str, Tensor]:
        """Return the current wavelet kernels for interpretation."""

        kernels = {}
        for idx, name in enumerate(self.wavelet_types):
            kernels[name] = self._wavelet_kernel(idx).detach().cpu()
        return kernels

