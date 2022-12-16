import copy
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from .reply_buffer import ReplyBuffer

class DDPG(nn.Module):
    def __init__(
        self, actor: nn.Module,
        critic: nn.Module,
        max_len: int,
        batch_size: int,
        random_process,
        tau: float = 0.001,
        gamma: float = 0.99,
        eps_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.memory = ReplyBuffer(max_len=max_len, batch_size=batch_size)
        self.random_process = random_process
        
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optim = T.optim.Adam(self.actor.parameters())

        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.critic_optim = T.optim.Adam(self.critic.parameters())

        self.tau = tau
        self.gamma = gamma
        self.epsilon = 1.0
        self.eps_decay = eps_decay
        
        self.to("cuda:0")

    def add(self, s, a, r, sn):
        self.memory.add(s, a, r, sn)

    def step(self):
        if not self.memory.can_sample():
            return
        s, a, r, ns = self.sample()

        next_q = self.critic_target(ns, self.actor_target(ns))
        target_q = r + self.gamma * next_q

        self.critic.zero_grad()
        q = self.critic(s, a)
        value_loss = F.mse_loss(q, target_q)
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss = (-self.critic(s, self.actor(s))).mean()
        policy_loss.backward()
        self.actor_optim.step()

        self.update(self.actor_target, self.actor)
        self.update(self.critic_target, self.critic)
        self.decay_eps()

    def sample(self):
        s, a, r, sn = self.memory.sample()
        s = T.Tensor(s).to("cuda:0")
        a = T.Tensor(a).to("cuda:0")
        r = T.Tensor(r).to("cuda:0")
        r = T.reshape(r, (self.memory.batch_size, 1))
        sn = T.Tensor(sn).to("cuda:0")
        return s, a, r, sn

    def update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def get_action(self, state: np.ndarray):
        s = np.array([state])
        s = T.Tensor(s).to("cuda:0")
        a = self.actor(s).cpu().detach().numpy()
        a += self.epsilon * self.random_process.sample()
        return a[0]
    
    def decay_eps(self):
        self.epsilon = self.epsilon * self.eps_decay
